"""

"""

from typing import Tuple

from rdagent.core.task import BaseTask, TaskLoader

# class data_ana: XXX


class Belief:
    """
    TODO: We may have better name for it.

    Name Candidates:
    - Hypothesis
    """

    # source: data_ana | model_nan = None


# Origin(path of repo/data/feedback) => view/summarization => generated Belief


class Scenario:
    def get_repo_path(self):
        """codebase"""

    def get_data(self):
        """ "data info"""

    def get_env(self):
        """env description"""


class Trace:
    scen: Scenario
    hist: list[Tuple[Belief, Feedback]]


class BeliefGen:
    def __init__(self, scen: Scenario):
        self.scen = scen

    def gen(self, trace: Trace) -> Belief:
        # def gen(self, scenario_desc: str, ) -> Belief:
        """
        Motivation of the variable `scenario_desc`:
        - Mocking a data-scientist is observing the scenario.

        scenario_desc may conclude:
        - data observation:
            - Original or derivative
        - Task information:
        """


class BeliefSet:
    """
    # drop, append

    belief_imp: list[float] | None  # importance of each belief
    failed_belief or success belief
    """

    belief_l: list[Belief]
    feedbacks: Dict[Tuple[Belief, Scenario], BeliefFeedback]


class Belief2Task(TaskLoader):
    """
    [Abstract description => conceret description] => Code implement
    """

    def convert(self, bs: BeliefSet) -> BaseTask:
        """Connect the idea proposal to implementation"""
        ...


class BeliefFeedback:
    ...


# Boolean, Reason, Confidence, etc.


class Imp2Feedback:
    """ "Generated(summarize) feedback from **Executed** Implemenation"""

    def summarize(self, ti: TaskImplementation) -> BeliefFeedback:
        """
        The `ti` should be exectued and the results should be included.
        For example: `mlflow` of Qlib will be included.
        """
