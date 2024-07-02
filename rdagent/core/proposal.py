"""

"""

from typing import Dict, List, Tuple

from rdagent.core.evolving_framework import Feedback
from rdagent.core.experiment import Experiment, Implementation, Loader, Task

# class data_ana: XXX


class Hypothesis:
    """
    TODO: We may have better name for it.

    Name Candidates:
    - Belief
    """

    hypothesis: str = None
    reason: str = None

    # source: data_ana | model_nan = None


# Origin(path of repo/data/feedback) => view/summarization => generated Hypothesis


class Scenario:
    def get_repo_path(self):
        """codebase"""

    def get_data(self):
        """ "data info"""

    def get_env(self):
        """env description"""

    def get_scenario_all_desc(self) -> str:
        """Combine all the description together"""


class HypothesisFeedback(Feedback): ...


class Trace:
    scen: Scenario
    hist: list[Tuple[Hypothesis, Experiment, HypothesisFeedback]]


class HypothesisGen:
    def __init__(self, scen: Scenario):
        self.scen = scen

    def gen(self, trace: Trace) -> Hypothesis:
        # def gen(self, scenario_desc: str, ) -> Hypothesis:
        """
        Motivation of the variable `scenario_desc`:
        - Mocking a data-scientist is observing the scenario.

        scenario_desc may conclude:
        - data observation:
            - Original or derivative
        - Task information:
        """


class HypothesisSet:
    """
    # drop, append

    hypothesis_imp: list[float] | None  # importance of each hypothesis
    true_hypothesis or false_hypothesis
    """

    hypothesis_list: list[Hypothesis]
    trace: Trace


class Hypothesis2Experiment(Loader[Experiment]):
    """
    [Abstract description => concrete description] => Code implement
    """

    def convert(self, bs: HypothesisSet) -> Experiment:
        """Connect the idea proposal to implementation"""
        ...


# Boolean, Reason, Confidence, etc.


class Experiment2Feedback:
    """ "Generated(summarize) feedback from **Executed** Implementation"""

    def summarize(self, ti: Experiment) -> HypothesisFeedback:
        """
        The `ti` should be executed and the results should be included.
        For example: `mlflow` of Qlib will be included.
        """
