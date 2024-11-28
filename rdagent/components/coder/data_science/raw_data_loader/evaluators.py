# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
    CoSTEERMultiFeedback
)
from rdagent.core.experiment import Task, Workspace
from rdagent.core.evolving_framework import QueriedKnowledge

class DataLoaderCoSTEEREvaluator(CoSTEEREvaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        