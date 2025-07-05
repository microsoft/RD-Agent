
from dev.feedback import DSExperiment2Feedback
from rdagent.core.scenario import Scenario

class DSDraftExperiment2Feedback(DSExperiment2Feedback):
    """
    A class to generate feedback for a DSExperiment based on the scenario and trace.
    This class extends the Experiment2Feedback class to provide specific functionality
    for data science experiments.
    """

    def __init__(self, scen: Scenario, version: str = "exp_feedback_v3") -> None:
        super().__init__(scen, version)
        self.version = version



