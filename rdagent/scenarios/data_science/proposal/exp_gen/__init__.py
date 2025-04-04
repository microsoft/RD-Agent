from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExpGen
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.draft import DSDraftExpGen
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import (
    DSProposalV1ExpGen,
    DSProposalV2ExpGen,
)
from rdagent.scenarios.data_science.scen import DataScienceScen


class DSExpGen(ExpGen):
    """Data Science Task Generator."""

    def __init__(self, scen: DataScienceScen, max_trace_hist: int = 3) -> None:
        self.max_trace_hist = max_trace_hist  # max number of historical trace to know when propose new experiment
        super().__init__(scen)

    def gen(self, trace: DSTrace) -> DSExperiment:
        if DS_RD_SETTING.coder_on_whole_pipeline:
            return DSProposalV2ExpGen(scen=self.scen).gen(
                trace=trace,
                max_trace_hist=self.max_trace_hist,
                pipeline=True,
            )
        next_missing_component = trace.next_incomplete_component()
        if next_missing_component is not None:
            return DSDraftExpGen(scen=self.scen).gen(
                component=next_missing_component,
                trace=trace,
            )
        if DS_RD_SETTING.proposal_version == "v1":
            return DSProposalV1ExpGen(scen=self.scen).gen(
                trace=trace,
                max_trace_hist=self.max_trace_hist,
            )
        if DS_RD_SETTING.proposal_version == "v2":
            return DSProposalV2ExpGen(scen=self.scen).gen(
                trace=trace,
                max_trace_hist=self.max_trace_hist,
            )
