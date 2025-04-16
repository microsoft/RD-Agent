import random

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import SOTAexpSelector, Trace
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment



class GlobalSOTASelector(SOTAexpSelector):
    """
    return the latest SOTA experiment from the trace to submit
    """

    def __init__(
        self,
    ):
        print(f"Using global SOTA policy by default")

    def get_sota_exp_to_submit(self, trace: Trace) -> DSExperiment | None:

        return trace.sota_experiment(search_type="all")

# TODO: more advanced sota exp selector (e.g. LLM-based, merge exp with multiple sub-trace)