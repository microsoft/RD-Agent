# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Task, Workspace


class DataLoaderCoSTEEREvaluator(CoSTEEREvaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return CoSTEERSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                shape_feedback="This task has failed too many times, skip implementation.",
                value_feedback="This task has failed too many times, skip implementation.",
                code_feedback="This task has failed too many times, skip implementation.",
                final_feedback="This task has failed too many times, skip implementation.",
                final_decision=False,
            )

        implementation.execute()

        return CoSTEERSingleFeedback()
