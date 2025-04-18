import random

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import SOTAexpSelector, Trace
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


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
    

class AutoSOTAexpSelector(SOTAexpSelector):
    """
    retrieve a list of SOTA experiments from the trace, then call the LLM to select the best one
    """

    def __init__(
        self,
    ):
        print(f"Using auto SOTA policy")

    def get_sota_exp_to_submit(self, trace: Trace) -> DSExperiment | None:
        # retrieve all SOTA experiments from the trace

        sota_exp_fb_list = trace.experiment_and_feedback_list_after_init(return_type="sota", search_type="all")
        score_list = []

        historical_attempts_with_scores_desc = "Historical proposal-evaluation analysis:\n\n"

        for exp, ef in sota_exp_fb_list:
            score = exp.result.loc["ensemble"].iloc[0]
            if score is not None:
                score_list.append(score)
        


# TODO: more advanced sota exp selector (e.g. LLM-based, merge exp with multiple sub-trace)
