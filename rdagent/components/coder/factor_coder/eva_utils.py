import io
import json
from abc import abstractmethod
from pathlib import Path
from typing import Tuple

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.experiment import Task, Workspace
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend

evaluate_prompts = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class FactorEvaluator:
    """Although the init method is same to Evaluator, but we want to emphasize they are different"""

    def __init__(self, scen=None) -> None:
        self.scen = scen

    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        **kwargs,
    ) -> Tuple[str, object]:
        """You can get the dataframe by

        .. code-block:: python

            _, gen_df = implementation.execute()
            _, gt_df = gt_implementation.execute()

        Returns
        -------
        Tuple[str, object]
            - str: the text-based description of the evaluation result
            - object: a comparable metric (bool, integer, float ...) None for evaluator with only text-based result

        """
        raise NotImplementedError("Please implement the `evaluator` method")

    def _get_df(self, gt_implementation: Workspace, implementation: Workspace):
        if gt_implementation is not None:
            _, gt_df = gt_implementation.execute()
            if isinstance(gt_df, pd.Series):
                gt_df = gt_df.to_frame("gt_factor")
            if isinstance(gt_df, pd.DataFrame):
                gt_df = gt_df.sort_index()
        else:
            gt_df = None

        _, gen_df = implementation.execute()
        if isinstance(gen_df, pd.Series):
            gen_df = gen_df.to_frame("source_factor")
        if isinstance(gen_df, pd.DataFrame):
            gen_df = gen_df.sort_index()
        return gt_df, gen_df

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorCodeEvaluator(FactorEvaluator):
    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        execution_feedback: str,
        value_feedback: str = "",
        gt_implementation: Workspace = None,
        **kwargs,
    ):
        factor_information = target_task.get_task_information()
        code = implementation.code

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(evaluate_prompts["evaluator_code_feedback_v1_system"])
            .render(
                scenario=(
                    self.scen.get_scenario_all_desc(
                        target_task,
                        filtered_tag="feature",
                        simple_background=FACTOR_COSTEER_SETTINGS.simple_background,
                    )
                    if self.scen is not None
                    else "No scenario description."
                )
            )
        )

        execution_feedback_to_render = execution_feedback
        for _ in range(10):  # 10 times to split the content is enough
            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    evaluate_prompts["evaluator_code_feedback_v1_user"],
                )
                .render(
                    factor_information=factor_information,
                    code=code,
                    execution_feedback=execution_feedback_to_render,
                    value_feedback=value_feedback,
                    gt_code=gt_implementation.code if gt_implementation else None,
                )
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                > LLM_SETTINGS.chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break
        critic_response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )

        return critic_response, None


class FactorInfEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        INF_count = gen_df.isin([float("inf"), -float("inf")]).sum().sum()
        if INF_count == 0:
            return "The source dataframe does not have any infinite values.", True
        else:
            return (
                f"The source dataframe has {INF_count} infinite values. Please check the implementation.",
                False,
            )


class FactorSingleColumnEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        if len(gen_df.columns) == 1:
            return "The source dataframe has only one column which is correct.", True
        else:
            return (
                "The source dataframe has more than one column. Please check the implementation. We only evaluate the first column.",
                False,
            )


class FactorOutputFormatEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Skip the evaluation of the output format.",
                False,
            )
        buffer = io.StringIO()
        gen_df.info(buf=buffer)
        gen_df_info_str = f"The user is currently working on a feature related task.\nThe output dataframe info is:\n{buffer.getvalue()}"
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(
                evaluate_prompts["evaluator_output_format_system"],
            )
            .render(
                scenario=(
                    self.scen.get_scenario_all_desc(implementation.target_task, filtered_tag="feature")
                    if self.scen is not None
                    else "No scenario description."
                )
            )
        )

        # TODO: with retry_context(retry_n=3, except_list=[KeyError]):
        max_attempts = 3
        attempts = 0
        final_evaluation_dict = None

        while attempts < max_attempts:
            try:
                api = APIBackend() if attempts == 0 else APIBackend(use_chat_cache=False)
                resp = api.build_messages_and_create_chat_completion(
                    user_prompt=gen_df_info_str, system_prompt=system_prompt, json_mode=True
                )
                resp_dict = json.loads(resp)
                resp_dict["output_format_decision"] = str(resp_dict["output_format_decision"]).lower() in ["true", "1"]

                return (
                    str(resp_dict["output_format_feedback"]),
                    resp_dict["output_format_decision"],
                )
            except (KeyError, json.JSONDecodeError) as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise KeyError(
                        "Wrong JSON Response or missing 'output_format_decision' or 'output_format_feedback' key after multiple attempts."
                    ) from e

        return "Failed to evaluate output format after multiple attempts.", False


class FactorDatetimeDailyEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str | object]:
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "The source dataframe is None. Skip the evaluation of the datetime format.", False

        if "datetime" not in gen_df.index.names:
            return "The source dataframe does not have a datetime index. Please check the implementation.", False

        try:
            pd.to_datetime(gen_df.index.get_level_values("datetime"))
        except Exception:
            return (
                f"The source dataframe has a datetime index but it is not in the correct format (maybe a regular string or other objects). Please check the implementation.\n The head of the output dataframe is: \n{gen_df.head()}",
                False,
            )

        time_diff = pd.to_datetime(gen_df.index.get_level_values("datetime")).to_series().diff().dropna().unique()
        if pd.Timedelta(minutes=1) in time_diff:
            return (
                "The generated dataframe is not daily. The implementation is definitely wrong. Please check the implementation.",
                False,
            )
        return "The generated dataframe is daily.", True


class FactorRowCountEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        ratio = min(len(gen_df), len(gt_df)) / max(len(gen_df), len(gt_df))
        return (
            (
                f"The ratio of rows count in the source dataframe to the ground truth dataframe is {ratio:.2f}. "
                + "Please verify the implementation. "
                if ratio <= 0.99
                else ""
            ),
            ratio,
        )


class FactorIndexEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        gen_index_set, gt_index_set = set(gen_df.index), set(gt_df.index)
        similarity = len(gen_index_set.intersection(gt_index_set)) / len(gen_index_set.union(gt_index_set))
        return (
            (
                f"The source dataframe and the ground truth dataframe have different index with a similarity of {similarity:.2%}. The similarity is calculated by the number of shared indices divided by the union indices. "
                + "Please check the implementation."
                if similarity <= 0.99
                else ""
            ),
            similarity,
        )


class FactorMissingValuesEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        if gen_df.isna().sum().sum() == gt_df.isna().sum().sum():
            return "Both dataframes have the same missing values.", True
        else:
            return (
                f"The dataframes do not have the same missing values. The source dataframe has {gen_df.isna().sum().sum()} missing values, while the ground truth dataframe has {gt_df.isna().sum().sum()} missing values. Please check the implementation.",
                False,
            )


class FactorEqualValueRatioEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                -1,
            )
        try:
            close_values = gen_df.sub(gt_df).abs().lt(1e-6)
            result_int = close_values.astype(int)
            pos_num = result_int.sum().sum()
            acc_rate = pos_num / close_values.size
        except:
            close_values = gen_df
        if close_values.all().iloc[0]:
            return (
                "All values in the dataframes are equal within the tolerance of 1e-6.",
                acc_rate,
            )
        else:
            return (
                "Some values differ by more than the tolerance of 1e-6. Check for rounding errors or differences in the calculation methods.",
                acc_rate,
            )


class FactorCorrelationEvaluator(FactorEvaluator):
    def __init__(self, hard_check: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hard_check = hard_check

    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        concat_df = pd.concat([gen_df, gt_df], axis=1)
        concat_df.columns = ["source", "gt"]
        ic = concat_df.groupby("datetime").apply(lambda df: df["source"].corr(df["gt"])).dropna().mean()
        ric = (
            concat_df.groupby("datetime")
            .apply(lambda df: df["source"].corr(df["gt"], method="spearman"))
            .dropna()
            .mean()
        )

        if self.hard_check:
            if ic > 0.99 and ric > 0.99:
                return (
                    f"The dataframes are highly correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}.",
                    True,
                )
            else:
                return (
                    f"The dataframes are not sufficiently high correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}. Investigate the factors that might be causing the discrepancies and ensure that the logic of the factor calculation is consistent.",
                    False,
                )
        else:
            return f"The ic is ({ic:.6f}) and the rankic is ({ric:.6f}).", ic


class FactorValueEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
        version: int = 1,  # 1 for qlib factors and 2 for kaggle factors
        **kwargs,
    ) -> Tuple:
        conclusions = []

        # Initialize result variables
        row_result = 0
        index_result = 0
        output_format_result = None
        equal_value_ratio_result = 0
        high_correlation_result = False
        row_result = None

        # Check if both dataframe has only one columns Mute this since factor task might generate more than one columns now
        if version == 1:
            feedback_str, _ = FactorSingleColumnEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)
        elif version == 2:
            input_shape = self.scen.input_shape
            _, gen_df = self._get_df(gt_implementation, implementation)
            if gen_df.shape[-1] > input_shape[-1]:
                conclusions.append(
                    "Output dataframe has more columns than input feature which is not acceptable in feature processing tasks. Please check the implementation to avoid generating too many columns. Consider this implementation as a failure."
                )

        feedback_str, inf_evaluate_res = FactorInfEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)

        # Check if the index of the dataframe is ("datetime", "instrument")
        feedback_str, _ = FactorOutputFormatEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)
        if version == 1:
            feedback_str, daily_check_result = FactorDatetimeDailyEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)
        else:
            daily_check_result = None

        # Check dataframe format
        if gt_implementation is not None:
            feedback_str, row_result = FactorRowCountEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            feedback_str, index_result = FactorIndexEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            feedback_str, output_format_result = FactorMissingValuesEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)

            feedback_str, equal_value_ratio_result = FactorEqualValueRatioEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)

            if index_result > 0.99:
                feedback_str, high_correlation_result = FactorCorrelationEvaluator(
                    hard_check=True, scen=self.scen
                ).evaluate(implementation, gt_implementation)
            else:
                high_correlation_result = False
                feedback_str = "The source dataframe and the ground truth dataframe have different index. Give up comparing the values and correlation because it's useless"
            conclusions.append(feedback_str)

        # Combine all conclusions into a single string
        conclusion_str = "\n".join(conclusions)

        if gt_implementation is not None and (equal_value_ratio_result > 0.99) or high_correlation_result:
            decision_from_value_check = True
        elif (
            row_result is not None
            and row_result <= 0.99
            or output_format_result is False
            or daily_check_result is False
            or inf_evaluate_res is False
        ):
            decision_from_value_check = False
        else:
            decision_from_value_check = None
        return conclusion_str, decision_from_value_check


class FactorFinalDecisionEvaluator(FactorEvaluator):
    def evaluate(
        self,
        target_task: FactorTask,
        execution_feedback: str,
        value_feedback: str,
        code_feedback: str,
        **kwargs,
    ) -> Tuple:
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(evaluate_prompts["evaluator_final_decision_v1_system"])
            .render(
                scenario=(
                    self.scen.get_scenario_all_desc(target_task, filtered_tag="feature")
                    if self.scen is not None
                    else "No scenario description."
                )
            )
        )
        execution_feedback_to_render = execution_feedback

        for _ in range(10):  # 10 times to split the content is enough
            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    evaluate_prompts["evaluator_final_decision_v1_user"],
                )
                .render(
                    factor_information=target_task.get_task_information(),
                    execution_feedback=execution_feedback_to_render,
                    code_feedback=code_feedback,
                    value_feedback=(
                        value_feedback
                        if value_feedback is not None
                        else "No Ground Truth Value provided, so no evaluation on value is performed."
                    ),
                )
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                > LLM_SETTINGS.chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break

        # TODO:  with retry_context(retry_n=3, except_list=[KeyError]):
        final_evaluation_dict = None
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            try:
                api = APIBackend() if attempts == 0 else APIBackend(use_chat_cache=False)
                final_evaluation_dict = json.loads(
                    api.build_messages_and_create_chat_completion(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_mode=True,
                        seed=attempts,  # in case of useless retrying when cache enabled.
                    ),
                )
                final_decision = final_evaluation_dict["final_decision"]
                final_feedback = final_evaluation_dict["final_feedback"]

                final_decision = str(final_decision).lower() in ["true", "1"]
                return final_decision, final_feedback

            except json.JSONDecodeError as e:
                raise ValueError("Failed to decode JSON response from API.") from e
            except KeyError as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise KeyError(
                        "Response from API is missing 'final_decision' or 'final_feedback' key after multiple attempts."
                    ) from e

        return None, None
