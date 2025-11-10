

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, ExperimentFeedback, Trace


class AgenticSysExp2Feedback(Experiment2Feedback):
    def generate_feedback(self, experiment: Experiment, trace: Trace) -> ExperimentFeedback:
        return ExperimentFeedback(
            reason="<This is a placeholder reason for the agentic system experiment>",
            decision=True,
        )
