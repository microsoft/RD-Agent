import re

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.factor_coder.eva_utils import (
    FactorCodeEvaluator,
    FactorFinalDecisionEvaluator,
    FactorValueEvaluator,
)
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Workspace

FactorSingleFeedback = CoSTEERSingleFeedback
FactorMultiFeedback = CoSTEERMultiFeedback


class FactorEvaluatorForCoder(CoSTEEREvaluator):
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
        implementation: Workspace,
        gt_implementation: Workspace = None,
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
                value_feedback="This task has failed too many times, skip value evaluation.",
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
                factor_feedback.value_feedback = "No factor value generated, skip value evaluation."
                factor_feedback.value_generated_flag = False
                decision_from_value_check = None
            else:
                factor_feedback.value_generated_flag = True
                (
                    factor_feedback.value_feedback,
                    decision_from_value_check,
                ) = self.value_evaluator.evaluate(
                    implementation=implementation, gt_implementation=gt_implementation, version=target_task.version
                )

            factor_feedback.final_decision_based_on_gt = gt_implementation is not None

            if decision_from_value_check is not None and decision_from_value_check is True:
                # To avoid confusion, when same_value_or_high_correlation is True, we do not need code feedback
                factor_feedback.code_feedback = "Final decision is True and there are no code critics."
                factor_feedback.final_decision = decision_from_value_check
                factor_feedback.final_feedback = "Value evaluation passed, skip final decision evaluation."
            elif decision_from_value_check is not None and decision_from_value_check is False:
                factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                    target_task=target_task,
                    implementation=implementation,
                    execution_feedback=factor_feedback.execution_feedback,
                    value_feedback=factor_feedback.value_feedback,
                    gt_implementation=gt_implementation,
                )
                factor_feedback.final_decision = decision_from_value_check
                factor_feedback.final_feedback = "Value evaluation failed, skip final decision evaluation."
            else:
                factor_feedback.code_feedback, _ = self.code_evaluator.evaluate(
                    target_task=target_task,
                    implementation=implementation,
                    execution_feedback=factor_feedback.execution_feedback,
                    value_feedback=factor_feedback.value_feedback,
                    gt_implementation=gt_implementation,
                )
                (
                    factor_feedback.final_decision,
                    factor_feedback.final_feedback,
                ) = self.final_decision_evaluator.evaluate(
                    target_task=target_task,
                    execution_feedback=factor_feedback.execution_feedback,
                    value_feedback=factor_feedback.value_feedback,
                    code_feedback=factor_feedback.code_feedback,
                )
            return factor_feedback


# TODO:
def shorten_prompt(tpl: str, render_kwargs: dict, shorten_key: str, max_trail: int = 10) -> str:
    """When the prompt is too long. We have to shorten it.
    But we should not truncate the prompt directly, so we should find the key we want to shorten and then shorten it.
    """
    # TODO: this should replace most of code in
    # - FactorFinalDecisionEvaluator.evaluate
    # - FactorCodeEvaluator.evaluate
