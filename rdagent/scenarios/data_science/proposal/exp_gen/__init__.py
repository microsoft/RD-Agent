from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExpGen
from rdagent.core.utils import import_class
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.draft import DSDraftExpGen
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import (
    DSProposalV1ExpGen,
    DSProposalV2ExpGen,
    DSProposalV3ExpGen,
)
from rdagent.scenarios.data_science.scen import DataScienceScen


class DSExpGen(ExpGen):
    """
    Data Science Task Generator.
    This is a experiment router generator;
    """

    def __init__(self, scen: DataScienceScen) -> None:
        super().__init__(scen)

    def gen(self, trace: DSTrace, selection: tuple[int, ...] = (-1,)) -> DSExperiment:

        # set the current selection for the trace
        # handy design:dynamically change the "current selection" attribute of the trace, and we donot need to pass selection as an argument to other functions
        trace.set_current_selection(selection)

        if DS_RD_SETTING.proposal_version not in ["v1", "v2", "v3"]:
            return import_class(DS_RD_SETTING.proposal_version)(scen=self.scen).gen(trace=trace)
        if DS_RD_SETTING.proposal_version == "v3":
            return DSProposalV3ExpGen(scen=self.scen).gen(trace=trace, pipeline=True)

        if DS_RD_SETTING.coder_on_whole_pipeline:
            return DSProposalV2ExpGen(scen=self.scen).gen(trace=trace, pipeline=True)

        next_missing_component = trace.next_incomplete_component()
        if next_missing_component is not None:
            return DSDraftExpGen(scen=self.scen).gen(
                component=next_missing_component,
                trace=trace,
            )
        if DS_RD_SETTING.proposal_version == "v1":
            return DSProposalV1ExpGen(scen=self.scen).gen(trace=trace)
        if DS_RD_SETTING.proposal_version == "v2":
            return DSProposalV2ExpGen(scen=self.scen).gen(trace=trace)
