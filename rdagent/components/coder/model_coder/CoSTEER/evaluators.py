from rdagent.core.evaluation import Evaluator
from rdagent.core.experiment import Implementation, Task


class ModelSingleFeedback:
    """This class is a feedback to single implementation which is generated from an evaluator."""

    def __init__(
        self,
        final_decision: bool = None,
        final_feedback: str = None,
    ) -> None:
        self.final_decision = final_decision
        self.final_feedback = final_feedback

    def __str__(self) -> str:
        return f"""------------------Model Final Feedback------------------
{self.final_feedback}
------------------Model Final Decision------------------
This implementation is {'SUCCESS' if self.final_decision else 'FAIL'}.
"""


class ModelEvaluator(Evaluator):
    def evaluate(self, target_task: Task, implementation: Implementation, gt_implementation: Implementation, **kwargs):
        return super().evaluate(target_task, implementation, gt_implementation, **kwargs)
