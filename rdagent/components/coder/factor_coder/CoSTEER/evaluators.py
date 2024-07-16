import io
import json
import re
from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_IMPLEMENT_SETTINGS
from rdagent.components.coder.factor_coder.CoSTEER.evolvable_subjects import (
    FactorEvolvingItem,
)
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Evaluator
from rdagent.core.evolving_framework import Feedback, QueriedKnowledge
from rdagent.core.experiment import Implementation, Task
from rdagent.log import rdagent_logger as logger
from rdagent.core.prompts import Prompts
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.oai.llm_utils import APIBackend

evaluate_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


class FactorEvaluator(Evaluator):
    # TODO:
    # I think we should have unified interface for all evaluates, for examples.
    # So we should adjust the interface of other factors
    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Implementation,
        gt_implementation: Implementation,
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

    def _get_df(self, gt_implementation: Implementation, implementation: Implementation):
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
        implementation: Implementation,
        execution_feedback: str,
        factor_value_feedback: str = "",
        gt_implementation: Implementation = None,
        **kwargs,
    ):
        factor_information = target_task.get_task_information()
        code = implementation.code

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(evaluate_prompts["evaluator_code_feedback_v1_system"])
            .render(scenario=self.scen.get_scenario_all_desc() if self.scen is not None else "No scenario description.")
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
                    factor_value_feedback=factor_value_feedback,
                    gt_code=gt_implementation.code if gt_implementation else None,
                )
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                > RD_AGENT_SETTINGS.chat_token_limit
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


class FactorSingleColumnEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Implementation,
        gt_implementation: Implementation,
    ) -> Tuple[str, object]:
        _, gen_df = self._get_df(gt_implementation, implementation)

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
        implementation: Implementation,
        gt_implementation: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Skip the evaluation of the output format.",
                False,
            )
        buffer = io.StringIO()
        gen_df.info(buf=buffer)
        gen_df_info_str = buffer.getvalue()
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(
                evaluate_prompts["evaluator_output_format_system"],
            )
            .render(scenario=self.scen.get_scenario_all_desc() if self.scen is not None else "No scenario description.")
        )
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=gen_df_info_str, system_prompt=system_prompt, json_mode=True
        )
        resp_dict = json.loads(resp)
        if isinstance(resp_dict["output_format_decision"], str) and resp_dict["output_format_decision"].lower() in (
            "true",
            "false",
        ):
            resp_dict["output_format_decision"] = bool(resp_dict["output_format_decision"])
        return (
            resp_dict["output_format_feedback"],
            resp_dict["output_format_decision"],
        )


class FactorDatetimeDailyEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Implementation,
        gt_implementation: Implementation,
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
                "The source dataframe has a datetime index but it is not in the correct format (maybe a regular string or other objects). Please check the implementation.",
                False,
            )

        time_diff = gen_df.index.get_level_values("datetime").to_series().diff().dropna().unique()
        if pd.Timedelta(minutes=1) in time_diff:
            return (
                "The generated dataframe is not daily. The implementation is definitely wrong. Please check the implementation.",
                False,
            )
        return "The generated dataframe is daily.", True


class FactorRowCountEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Implementation,
        gt_implementation: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)

        if gen_df.shape[0] == gt_df.shape[0]:
            return "Both dataframes have the same rows count.", True
        else:
            return (
                f"The source dataframe and the ground truth dataframe have different rows count. The source dataframe has {gen_df.shape[0]} rows, while the ground truth dataframe has {gt_df.shape[0]} rows. Please check the implementation.",
                False,
            )


class FactorIndexEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Implementation,
        gt_implementation: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)

        if gen_df.index.equals(gt_df.index):
            return "Both dataframes have the same index.", True
        else:
            return (
                "The source dataframe and the ground truth dataframe have different index. Please check the implementation.",
                False,
            )


class FactorMissingValuesEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Implementation,
        gt_implementation: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)

        if gen_df.isna().sum().sum() == gt_df.isna().sum().sum():
            return "Both dataframes have the same missing values.", True
        else:
            return (
                f"The dataframes do not have the same missing values. The source dataframe has {gen_df.isna().sum().sum()} missing values, while the ground truth dataframe has {gt_df.isna().sum().sum()} missing values. Please check the implementation.",
                False,
            )


class FactorEqualValueCountEvaluator(FactorEvaluator):
    def evaluate(
        self,
        implementation: Implementation,
        gt_implementation: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)

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
        implementation: Implementation,
        gt_implementation: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt_implementation, implementation)

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
        implementation: Implementation,
        gt_implementation: Implementation,
        **kwargs,
    ) -> Tuple:
        conclusions = []

        # Check if both dataframe has only one columns
        feedback_str, _ = FactorSingleColumnEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)

        # Check if the index of the dataframe is ("datetime", "instrument")
        feedback_str, _ = FactorOutputFormatEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)

        feedback_str, daily_check_result = FactorDatetimeDailyEvaluator(self.scen).evaluate(
            implementation, gt_implementation
        )
        conclusions.append(feedback_str)

        # Check if both dataframe have the same rows count
        if gt_implementation is not None:
            feedback_str, _ = FactorRowCountEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            feedback_str, same_index_result = FactorIndexEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)

            feedback_str, _ = FactorMissingValuesEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            feedback_str, equal_value_ratio_result = FactorEqualValueCountEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)

            if same_index_result:
                feedback_str, high_correlation_result = FactorCorrelationEvaluator(
                    hard_check=True, scen=self.scen
                ).evaluate(implementation, gt_implementation)
            else:
                high_correlation_result = False
                feedback_str = "The source dataframe and the ground truth dataframe have different index. Give up comparing the values and correlation because it's useless"
            conclusions.append(feedback_str)
        else:
            equal_value_ratio_result = 0
            high_correlation_result = False

        # Combine all conclusions into a single string
        conclusion_str = "\n".join(conclusions)

        if gt_implementation is not None and (equal_value_ratio_result > 0.99) or high_correlation_result:
            decision_from_value_check = True
        elif daily_check_result is False:
            decision_from_value_check = False
        else:
            decision_from_value_check = None
        return conclusion_str, decision_from_value_check


class FactorFinalDecisionEvaluator(Evaluator):
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
            .render(scenario=self.scen.get_scenario_all_desc() if self.scen is not None else "No scenario description.")
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
                    factor_value_feedback=(
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
                > RD_AGENT_SETTINGS.chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break

        final_evaluation_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            ),
        )
        if isinstance(final_evaluation_dict["final_decision"], str) and final_evaluation_dict[
            "final_decision"
        ].lower() in ("true", "false"):
            final_evaluation_dict["final_decision"] = bool(final_evaluation_dict["final_decision"])
        return (
            final_evaluation_dict["final_decision"],
            final_evaluation_dict["final_feedback"],
        )


class FactorSingleFeedback:
    """This class is a feedback to single implementation which is generated from an evaluator."""

    def __init__(
        self,
        execution_feedback: str = None,
        value_generated_flag: bool = False,
        code_feedback: str = None,
        factor_value_feedback: str = None,
        final_decision: bool = None,
        final_feedback: str = None,
        final_decision_based_on_gt: bool = None,
    ) -> None:
        self.execution_feedback = execution_feedback
        self.value_generated_flag = value_generated_flag
        self.code_feedback = code_feedback
        self.factor_value_feedback = factor_value_feedback
        self.final_decision = final_decision
        self.final_feedback = final_feedback
        self.final_decision_based_on_gt = final_decision_based_on_gt

    def __str__(self) -> str:
        return f"""------------------Factor Execution Feedback------------------
{self.execution_feedback}
------------------Factor Code Feedback------------------
{self.code_feedback}
------------------Factor Value Feedback------------------
{self.factor_value_feedback}
------------------Factor Final Feedback------------------
{self.final_feedback}
------------------Factor Final Decision------------------
This implementation is {'SUCCESS' if self.final_decision else 'FAIL'}.
"""


class FactorMultiFeedback(
    Feedback,
    List[FactorSingleFeedback],
):
    """Feedback contains a list, each element is the corresponding feedback for each factor implementation."""


class FactorEvaluatorForCoder(FactorEvaluator):
    """This class is the v1 version of evaluator for a single factor implementation.
    It calls several evaluators in share modules to evaluate the factor implementation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value_evaluator = FactorValueEvaluator(self.scen)
        self.code_evaluator = FactorCodeEvaluator(self.scen)
        self.final_decision_evaluator = FactorFinalDecisionEvaluator(self.scen)

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Implementation,
        gt_implementation: Implementation = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorSingleFeedback:
        if implementation is None:
            return None

        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return FactorSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                value_generated_flag=False,
                code_feedback="This task has failed too many times, skip code evaluation.",
                factor_value_feedback="This task has failed too many times, skip value evaluation.",
                final_decision=False,
                final_feedback="This task has failed too many times, skip final decision evaluation.",
                final_decision_based_on_gt=False,
            )
        else:
            factor_feedback = FactorSingleFeedback()

            # 1. Get factor execution feedback to generated implementation and remove the long list of numbers in execution feedback
            (
                execution_feedback,
                gen_df,
            ) = implementation.execute()

            execution_feedback = re.sub(r"(?<=\D)(,\s+-?\d+\.\d+){50,}(?=\D)", ", ", execution_feedback)
            factor_feedback.execution_feedback = "\n".join(
                [line for line in execution_feedback.split("\n") if "warning" not in line.lower()]
            )

            # 2. Get factor value feedback
            if gen_df is None:
                factor_feedback.factor_value_feedback = "No factor value generated, skip value evaluation."
                factor_feedback.value_generated_flag = False
                decision_from_value_check = None
            else:
                factor_feedback.value_generated_flag = True
                (
                    factor_feedback.factor_value_feedback,
                    decision_from_value_check,
                ) = self.value_evaluator.evaluate(implementation=implementation, gt_implementation=gt_implementation)

            factor_feedback.final_decision_based_on_gt = gt_implementation is not None

            if decision_from_value_check is not None and decision_from_value_check is True:
                # To avoid confusion, when same_value_or_high_correlation is True, we do not need code feedback
                factor_feedback.code_feedback = "Final decision is True and there are no code critics."
                factor_feedback.final_decision = decision_from_value_check
                factor_feedback.final_feedback = "Value evaluation passed, skip final decision evaluation."
            elif decision_from_value_check is not None and decision_from_value_check is False:
                factor_feedback.code_feedback = (
                    "Final decision is False because value evaluation gets a confident rejection to the result."
                )
                factor_feedback.final_decision = decision_from_value_check
                factor_feedback.final_feedback = "Value evaluation failed, skip final decision evaluation."
            else:
                factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                    target_task=target_task,
                    implementation=implementation,
                    execution_feedback=factor_feedback.execution_feedback,
                    value_feedback=factor_feedback.factor_value_feedback,
                    gt_implementation=gt_implementation,
                )
                (
                    factor_feedback.final_decision,
                    factor_feedback.final_feedback,
                ) = self.final_decision_evaluator.evaluate(
                    target_task=target_task,
                    execution_feedback=factor_feedback.execution_feedback,
                    value_feedback=factor_feedback.factor_value_feedback,
                    code_feedback=factor_feedback.code_feedback,
                )
            logger.info(factor_feedback.final_decision)
            return factor_feedback


class FactorMultiEvaluator(Evaluator):
    def __init__(self, single_evaluator, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.single_factor_implementation_evaluator = single_evaluator

    def evaluate(
        self,
        evo: FactorEvolvingItem,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorMultiFeedback:
        multi_implementation_feedback = FactorMultiFeedback()

        # for index in range(len(evo.sub_tasks)):
        #     corresponding_implementation = evo.sub_implementations[index]
        #     corresponding_gt_implementation = (
        #         evo.sub_gt_implementations[index] if evo.sub_gt_implementations is not None else None
        #     )

        #     multi_implementation_feedback.append(
        #         self.single_factor_implementation_evaluator.evaluate(
        #             target_task=evo.sub_tasks[index],
        #             implementation=corresponding_implementation,
        #             gt_implementation=corresponding_gt_implementation,
        #             queried_knowledge=queried_knowledge,
        #         )
        #     )

        calls = []
        for index in range(len(evo.sub_tasks)):
            corresponding_implementation = evo.sub_implementations[index]
            corresponding_gt_implementation = (
                evo.sub_gt_implementations[index] if evo.sub_gt_implementations is not None else None
            )
            calls.append(
                (
                    self.single_factor_implementation_evaluator.evaluate,
                    (
                        evo.sub_tasks[index],
                        corresponding_implementation,
                        corresponding_gt_implementation,
                        queried_knowledge,
                    ),
                ),
            )
        multi_implementation_feedback = multiprocessing_wrapper(calls, n=FACTOR_IMPLEMENT_SETTINGS.evo_multi_proc_n)

        final_decision = [
            None if single_feedback is None else single_feedback.final_decision
            for single_feedback in multi_implementation_feedback
        ]
        logger.info(f"Final decisions: {final_decision} True count: {final_decision.count(True)}")

        return multi_implementation_feedback


# TODO:
def shorten_prompt(tpl: str, render_kwargs: dict, shorten_key: str, max_trail: int = 10) -> str:
    """When the prompt is too long. We have to shorten it.
    But we should not truncate the prompt directly, so we should find the key we want to shorten and then shorten it.
    """
    # TODO: this should replace most of code in
    # - FactorFinalDecisionEvaluator.evaluate
    # - FactorCodeEvaluator.evaluate
