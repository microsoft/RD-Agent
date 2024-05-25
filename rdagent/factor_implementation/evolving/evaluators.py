from __future__ import annotations

import re
from typing import List

from pandas.core.api import DataFrame as DataFrame

from core.evolving_framework import Evaluator as EvolvingEvaluator
from core.evolving_framework import Feedback, QueriedKnowledge
from core.log import FinCoLog
from core.utils import multiprocessing_wrapper
from factor_implementation.evolving.evolvable_subjects import (
    FactorImplementationList,
)
from factor_implementation.share_modules.conf import FactorImplementSettings
from factor_implementation.share_modules.evaluator import (
    Evaluator as FactorImplementationEvaluator,
)
from factor_implementation.share_modules.evaluator import (
    FactorImplementationCodeEvaluator,
    FactorImplementationFinalDecisionEvaluator,
    FactorImplementationValueEvaluator,
)
from factor_implementation.share_modules.factor import (
    FactorImplementation,
    FactorImplementationTask,
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
        target_task: FactorImplementationTask,
        implementation: FactorImplementation,
        gt_implementation: FactorImplementation = None,
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
                    FinCoLog().warning("Value evaluation failed with exception: %s", e)
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


class FactorImplementationsMultiEvaluator(EvolvingEvaluator):
    def __init__(self, single_evaluator=FactorImplementationEvaluatorV1()) -> None:
        super().__init__()
        self.single_factor_implementation_evaluator = single_evaluator

    def evaluate(
        self,
        evo: FactorImplementationList,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorImplementationsMultiFeedback:
        multi_implementation_feedback = FactorImplementationsMultiFeedback()

        # for index in range(len(evo.target_factor_tasks)):
        #     corresponding_implementation = evo.corresponding_implementations[index]
        #     corresponding_gt_implementation = (
        #         evo.corresponding_gt_implementations[index]
        #         if evo.corresponding_gt_implementations is not None
        #         else None
        #     )

        #     multi_implementation_feedback.append(
        #         self.single_factor_implementation_evaluator.evaluate(
        #             target_task=evo.target_factor_tasks[index],
        #             implementation=corresponding_implementation,
        #             gt_implementation=corresponding_gt_implementation,
        #             queried_knowledge=queried_knowledge,
        #         )
        #     )

        calls = []
        for index in range(len(evo.target_factor_tasks)):
            corresponding_implementation = evo.corresponding_implementations[index]
            corresponding_gt_implementation = (
                evo.corresponding_gt_implementations[index]
                if evo.corresponding_gt_implementations is not None
                else None
            )
            calls.append(
                (
                    self.single_factor_implementation_evaluator.evaluate,
                    (
                        evo.target_factor_tasks[index],
                        corresponding_implementation,
                        corresponding_gt_implementation,
                        queried_knowledge,
                    ),
                ),
            )
        multi_implementation_feedback = multiprocessing_wrapper(calls, n=FactorImplementSettings().evo_multi_proc_n)

        final_decision = [
            None if single_feedback is None else single_feedback.final_decision
            for single_feedback in multi_implementation_feedback
        ]
        print(f"Final decisions: {final_decision} True count: {final_decision.count(True)}")

        return multi_implementation_feedback
