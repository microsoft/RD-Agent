from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExpGen
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.draft.draft_v1 import DSDraftExpGenV2
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import DSProposalV2ExpGen


class DraftRouterExpGenV1(ExpGen):
    """
    A intelligent router for drafting and improving.

    """
    def __init__(self, *args, **kwargs):
        self.draft_exp_gen = DSDraftExpGenV2(self.scen)
        self.base_exp_gen = DSProposalV2ExpGen(self.scen)
        # In the future,
        # self.draft_exp_gen = ImproveExpGen(self.scen)


    def gen(self, trace: DSTrace) -> DSExperiment:
        pipeline = DS_RD_SETTING.coder_on_whole_pipeline
        sota_exp = trace.sota_experiment()
        if sota_exp is None and pipeline:
            return self.draft_exp_gen.gen(trace)
        return self.base_exp_gen.gen(trace)
    

class DraftRouterExpGenV2(ExpGen):
    """
    A intelligent router for drafting and improving.

    """
    def __init__(self, *args, **kwargs):
        self.draft_exp_gen = DSDraftExpGenV2(self.scen)
        self.base_exp_gen = DSProposalV2ExpGen(self.scen)
        # In the future,
        # self.draft_exp_gen = ImproveExpGen(self.scen)


    def gen(self, trace: DSTrace) -> DSExperiment:
        pipeline = DS_RD_SETTING.coder_on_whole_pipeline
        sota_exp = trace.sota_experiment()
        if sota_exp is None and pipeline:
            return self.draft_exp_gen.gen(trace)
        return self.base_exp_gen.gen(trace)


"""
# Default
DS_HYPOTHESIS_GEN="rdagent.scenarios.data_science.proposal.exp_gen.proposal.DSProposalV2ExpGen"

# Your compared method
DS_HYPOTHESIS_GEN="rdagent.scenarios.data_science.proposal.exp_gen.proposal.router.DraftRouterExpGenV1"
"""