import json
import re
from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.task_implementation.factor_implementation.config import (
    FACTOR_IMPLEMENT_SETTINGS,
)
from rdagent.components.task_implementation.factor_implementation.evolving.evolvable_subjects import (
    FactorEvolvingItem,
)
from rdagent.components.task_implementation.factor_implementation.evolving.evolving_strategy import (
    FactorTask,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Evaluator
from rdagent.core.evolving_framework import Feedback, QueriedKnowledge
from rdagent.core.experiment import Implementation
from rdagent.core.log import RDAgentLog
from rdagent.core.prompts import Prompts
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.oai.llm_utils import APIBackend

evaluate_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


class FactorImplementationEvaluator(Evaluator):
    # TODO:
    # I think we should have unified interface for all evaluates, for examples.
    # So we should adjust the interface of other factors
    @abstractmethod
    def evaluate(
        self,
        gt: Implementation,
        gen: Implementation,
    ) -> Tuple[str, object]:
        """You can get the dataframe by

        .. code-block:: python

            _, gt_df = gt.execute()
            _, gen_df = gen.execute()

        Returns
        -------
        Tuple[str, object]
            - str: the text-based description of the evaluation result
            - object: a comparable metric (bool, integer, float ...)

        """
        raise NotImplementedError("Please implement the `evaluator` method")

    def _get_df(self, gt: Implementation, gen: Implementation):
        _, gt_df = gt.execute()
        _, gen_df = gen.execute()
        if isinstance(gen_df, pd.Series):
            gen_df = gen_df.to_frame("source_factor")
        if isinstance(gt_df, pd.Series):
            gt_df = gt_df.to_frame("gt_factor")
        return gt_df, gen_df


class FactorImplementationCodeEvaluator(Evaluator):
    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Implementation,
        execution_feedback: str,
        factor_value_feedback: str = "",
        gt_implementation: Implementation = None,
        **kwargs,
    ):
        factor_information = target_task.get_factor_information()
        code = implementation.code

        system_prompt = evaluate_prompts["evaluator_code_feedback_v1_system"]

        execution_feedback_to_render = execution_feedback
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
        while (
            APIBackend().build_messages_and_calculate_token(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                former_messages=[],
            )
            > RD_AGENT_SETTINGS.chat_token_limit
        ):
            execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
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
        critic_response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )

        return critic_response


class FactorImplementationSingleColumnEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: Implementation,
        gen: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        if len(gen_df.columns) == 1 and len(gt_df.columns) == 1:
            return "Both dataframes have only one column.", True
        elif len(gen_df.columns) != 1:
            gen_df = gen_df.iloc(axis=1)[
                [
                    0,
                ]
            ]
            return (
                "The source dataframe has more than one column. Please check the implementation. We only evaluate the first column.",
                False,
            )
        return "", False

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationIndexFormatEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: Implementation,
        gen: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)
        idx_name_right = gen_df.index.names == ("datetime", "instrument")
        if idx_name_right:
            return (
                'The index of the dataframe is ("datetime", "instrument") and align with the predefined format.',
                True,
            )
        else:
            return (
                'The index of the dataframe is not ("datetime", "instrument"). Please check the implementation.',
                False,
            )

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationRowCountEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: Implementation,
        gen: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        if gen_df.shape[0] == gt_df.shape[0]:
            return "Both dataframes have the same rows count.", True
        else:
            return (
                f"The source dataframe and the ground truth dataframe have different rows count. The source dataframe has {gen_df.shape[0]} rows, while the ground truth dataframe has {gt_df.shape[0]} rows. Please check the implementation.",
                False,
            )

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationIndexEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: Implementation,
        gen: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        if gen_df.index.equals(gt_df.index):
            return "Both dataframes have the same index.", True
        else:
            return (
                "The source dataframe and the ground truth dataframe have different index. Please check the implementation.",
                False,
            )

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationMissingValuesEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: Implementation,
        gen: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

        if gen_df.isna().sum().sum() == gt_df.isna().sum().sum():
            return "Both dataframes have the same missing values.", True
        else:
            return (
                f"The dataframes do not have the same missing values. The source dataframe has {gen_df.isna().sum().sum()} missing values, while the ground truth dataframe has {gt_df.isna().sum().sum()} missing values. Please check the implementation.",
                False,
            )

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationValuesEvaluator(FactorImplementationEvaluator):
    def evaluate(
        self,
        gt: Implementation,
        gen: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

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

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationCorrelationEvaluator(FactorImplementationEvaluator):
    def __init__(self, hard_check: bool) -> None:
        self.hard_check = hard_check

    def evaluate(
        self,
        gt: Implementation,
        gen: Implementation,
    ) -> Tuple[str, object]:
        gt_df, gen_df = self._get_df(gt, gen)

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

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationValEvaluator(FactorImplementationEvaluator):
    def evaluate(self, gt: Implementation, gen: Implementation):
        _, gt_df = gt.execute()
        _, gen_df = gen.execute()
        # FIXME: refactor the two classes
        fiv = FactorImplementationValueEvaluator()
        return fiv.evaluate(source_df=gen_df, gt_df=gt_df)

    def __str__(self) -> str:
        return self.__class__.__name__


class FactorImplementationValueEvaluator(Evaluator):
    # TODO: let's discuss the about the interface of the evaluator
    def evaluate(
        self,
        source_df: pd.DataFrame,
        gt_df: pd.DataFrame,
        **kwargs,
    ) -> Tuple:
        conclusions = []

        if isinstance(source_df, pd.Series):
            source_df = source_df.to_frame("source_factor")
            conclusions.append(
                "The source dataframe is a series, better convert it to a dataframe.",
            )
        if gt_df is not None and isinstance(gt_df, pd.Series):
            gt_df = gt_df.to_frame("gt_factor")
            conclusions.append(
                "The ground truth dataframe is a series, convert it to a dataframe.",
            )

        # Check if both dataframe has only one columns
        if len(source_df.columns) == 1:
            conclusions.append("The source dataframe has only one column which is correct.")
        else:
            conclusions.append(
                "The source dataframe has more than one column. Please check the implementation. We only evaluate the first column.",
            )
            source_df = source_df.iloc(axis=1)[
                [
                    0,
                ]
            ]

        if list(source_df.index.names) != ["datetime", "instrument"]:
            conclusions.append(
                rf"The index of the dataframe is not (\"datetime\", \"instrument\"), instead is {source_df.index.names}. Please check the implementation.",
            )
        else:
            conclusions.append(
                'The index of the dataframe is ("datetime", "instrument") and align with the predefined format.',
            )

        # Check if both dataframe have the same rows count
        if gt_df is not None:
            if source_df.shape[0] == gt_df.shape[0]:
                conclusions.append("Both dataframes have the same rows count.")
                same_row_count_result = True
            else:
                conclusions.append(
                    f"The source dataframe and the ground truth dataframe have different rows count. The source dataframe has {source_df.shape[0]} rows, while the ground truth dataframe has {gt_df.shape[0]} rows. Please check the implementation.",
                )
                same_row_count_result = False

            # Check whether both dataframe has the same index
            if source_df.index.equals(gt_df.index):
                conclusions.append("Both dataframes have the same index.")
                same_index_result = True
            else:
                conclusions.append(
                    "The source dataframe and the ground truth dataframe have different index. Please check the implementation.",
                )
                same_index_result = False

            # Check for the same missing values (NaN)
            if source_df.isna().sum().sum() == gt_df.isna().sum().sum():
                conclusions.append("Both dataframes have the same missing values.")
                same_missing_values_result = True
            else:
                conclusions.append(
                    f"The dataframes do not have the same missing values. The source dataframe has {source_df.isna().sum().sum()} missing values, while the ground truth dataframe has {gt_df.isna().sum().sum()} missing values. Please check the implementation.",
                )
                same_missing_values_result = False

            # Check if the values are the same within a small tolerance
            if not same_index_result:
                conclusions.append(
                    "The source dataframe and the ground truth dataframe have different index. Give up comparing the values and correlation because it's useless",
                )
                same_values_result = False
                high_correlation_result = False
            else:
                close_values = source_df.sub(gt_df).abs().lt(1e-6)
                if close_values.all().iloc[0]:
                    conclusions.append(
                        "All values in the dataframes are equal within the tolerance of 1e-6.",
                    )
                    same_values_result = True
                else:
                    conclusions.append(
                        "Some values differ by more than the tolerance of 1e-6. Check for rounding errors or differences in the calculation methods.",
                    )
                    same_values_result = False

                # Check the ic and rankic between the two dataframes
                concat_df = pd.concat([source_df, gt_df], axis=1)
                concat_df.columns = ["source", "gt"]
                try:
                    ic = concat_df.groupby("datetime").apply(lambda df: df["source"].corr(df["gt"])).dropna().mean()
                    ric = (
                        concat_df.groupby("datetime")
                        .apply(lambda df: df["source"].corr(df["gt"], method="spearman"))
                        .dropna()
                        .mean()
                    )

                    if ic > 0.99 and ric > 0.99:
                        conclusions.append(
                            f"The dataframes are highly correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}.",
                        )
                        high_correlation_result = True
                    else:
                        conclusions.append(
                            f"The dataframes are not sufficiently high correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}. Investigate the factors that might be causing the discrepancies and ensure that the logic of the factor calculation is consistent.",
                        )
                        high_correlation_result = False

                    # Check for shifted alignments only in the "datetime" index
                    max_shift_days = 2
                    for shift in range(-max_shift_days, max_shift_days + 1):
                        if shift == 0:
                            continue  # Skip the case where there is no shift

                        shifted_source_df = source_df.groupby(level="instrument").shift(shift)
                        concat_df = pd.concat([shifted_source_df, gt_df], axis=1)
                        concat_df.columns = ["source", "gt"]
                        shifted_ric = (
                            concat_df.groupby("datetime")
                            .apply(lambda df: df["source"].corr(df["gt"], method="spearman"))
                            .dropna()
                            .mean()
                        )
                        if shifted_ric > 0.99:
                            conclusions.append(
                                f"The dataframes are highly correlated with a shift of {max_shift_days} days in the 'date' index. Shifted rankic: {shifted_ric:.6f}.",
                            )
                            break
                    else:
                        conclusions.append(
                            f"No sufficient correlation found when shifting up to {max_shift_days} days in the 'date' index. Investigate the factors that might be causing discrepancies.",
                        )

                except Exception as e:
                    RDAgentLog().warning(f"Error occurred when calculating the correlation: {str(e)}")
                    conclusions.append(
                        f"Some error occurred when calculating the correlation. Investigate the factors that might be causing the discrepancies and ensure that the logic of the factor calculation is consistent. Error: {e}",
                    )
                    high_correlation_result = False

        # Combine all conclusions into a single string
        conclusion_str = "\n".join(conclusions)

        final_result = (same_values_result or high_correlation_result) if gt_df is not None else False
        return conclusion_str, final_result


# TODO:
def shorten_prompt(tpl: str, render_kwargs: dict, shorten_key: str, max_trail: int = 10) -> str:
    """When the prompt is too long. We have to shorten it.
    But we should not truncate the prompt directly, so we should find the key we want to shorten and then shorten it.
    """
    # TODO: this should replace most of code in
    # - FactorImplementationFinalDecisionEvaluator.evaluate
    # - FactorImplementationCodeEvaluator.evaluate


class FactorImplementationFinalDecisionEvaluator(Evaluator):
    def evaluate(
        self,
        target_task: FactorTask,
        execution_feedback: str,
        value_feedback: str,
        code_feedback: str,
        **kwargs,
    ) -> Tuple:
        system_prompt = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")[
            "evaluator_final_decision_v1_system"
        ]
        execution_feedback_to_render = execution_feedback
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(
                evaluate_prompts["evaluator_final_decision_v1_user"],
            )
            .render(
                factor_information=target_task.get_factor_information(),
                execution_feedback=execution_feedback_to_render,
                code_feedback=code_feedback,
                factor_value_feedback=(
                    value_feedback
                    if value_feedback is not None
                    else "No Ground Truth Value provided, so no evaluation on value is performed."
                ),
            )
        )
        while (
            APIBackend().build_messages_and_calculate_token(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                former_messages=[],
            )
            > RD_AGENT_SETTINGS.chat_token_limit
        ):
            execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    evaluate_prompts["evaluator_final_decision_v1_user"],
                )
                .render(
                    factor_information=target_task.get_factor_information(),
                    execution_feedback=execution_feedback_to_render,
                    code_feedback=code_feedback,
                    factor_value_feedback=(
                        value_feedback
                        if value_feedback is not None
                        else "No Ground Truth Value provided, so no evaluation on value is performed."
                    ),
                )
            )

        final_evaluation_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            ),
        )
        return (
            final_evaluation_dict["final_decision"],
            final_evaluation_dict["final_feedback"],
        )


class FactorImplementationSingleFeedback:
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


class FactorImplementationsMultiFeedback(
    Feedback,
    List[FactorImplementationSingleFeedback],
):
    """Feedback contains a list, each element is the corresponding feedback for each factor implementation."""


class FactorImplementationEvaluatorV1(FactorImplementationEvaluator):
    """This class is the v1 version of evaluator for a single factor implementation.
    It calls several evaluators in share modules to evaluate the factor implementation.
    """

    def __init__(self) -> None:
        self.code_evaluator = FactorImplementationCodeEvaluator()
        self.value_evaluator = FactorImplementationValueEvaluator()
        self.final_decision_evaluator = FactorImplementationFinalDecisionEvaluator()

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Implementation,
        gt_implementation: Implementation = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorImplementationSingleFeedback:
        if implementation is None:
            return None

        target_task_information = target_task.get_factor_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return FactorImplementationSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                value_generated_flag=False,
                code_feedback="This task has failed too many times, skip code evaluation.",
                factor_value_feedback="This task has failed too many times, skip value evaluation.",
                final_decision=False,
                final_feedback="This task has failed too many times, skip final decision evaluation.",
                final_decision_based_on_gt=False,
            )
        else:
            factor_feedback = FactorImplementationSingleFeedback()
            (
                factor_feedback.execution_feedback,
                source_df,
            ) = implementation.execute()

            # Remove the long list of numbers in the feedback
            pattern = r"(?<=\D)(,\s+-?\d+\.\d+){50,}(?=\D)"
            factor_feedback.execution_feedback = re.sub(pattern, ", ", factor_feedback.execution_feedback)
            execution_feedback_lines = [
                line for line in factor_feedback.execution_feedback.split("\n") if "warning" not in line.lower()
            ]
            factor_feedback.execution_feedback = "\n".join(execution_feedback_lines)

            if source_df is None:
                factor_feedback.factor_value_feedback = "No factor value generated, skip value evaluation."
                factor_feedback.value_generated_flag = False
                value_decision = None
            else:
                factor_feedback.value_generated_flag = True
                if gt_implementation is not None:
                    _, gt_df = gt_implementation.execute(store_result=True)
                else:
                    gt_df = None
                try:
                    source_df = source_df.sort_index()
                    if gt_df is not None:
                        gt_df = gt_df.sort_index()
                    (
                        factor_feedback.factor_value_feedback,
                        value_decision,
                    ) = self.value_evaluator.evaluate(source_df=source_df, gt_df=gt_df)
                except Exception as e:
                    RDAgentLog().warning("Value evaluation failed with exception: %s", e)
                    factor_feedback.factor_value_feedback = "Value evaluation failed."
                    value_decision = False

            factor_feedback.final_decision_based_on_gt = gt_implementation is not None

            if value_decision is not None and value_decision is True:
                # To avoid confusion, when value_decision is True, we do not need code feedback
                factor_feedback.code_feedback = "Final decision is True and there are no code critics."
                factor_feedback.final_decision = value_decision
                factor_feedback.final_feedback = "Value evaluation passed, skip final decision evaluation."
            else:
                factor_feedback.code_feedback = self.code_evaluator.evaluate(
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
            return factor_feedback


class FactorImplementationsMultiEvaluator(Evaluator):
    def __init__(self, single_evaluator=FactorImplementationEvaluatorV1()) -> None:
        super().__init__()
        self.single_factor_implementation_evaluator = single_evaluator

    def evaluate(
        self,
        evo: FactorEvolvingItem,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorImplementationsMultiFeedback:
        multi_implementation_feedback = FactorImplementationsMultiFeedback()

        # for index in range(len(evo.sub_tasks)):
        #     corresponding_implementation = evo.sub_implementations[index]
        #     corresponding_gt_implementation = (
        #         evo.sub_gt_implementations[index]
        #         if evo.sub_gt_implementations is not None
        #         else None
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
        RDAgentLog().info(f"Final decisions: {final_decision} True count: {final_decision.count(True)}")

        return multi_implementation_feedback
