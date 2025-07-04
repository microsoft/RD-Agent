from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExpGen
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.draft import DSDraftExpGen
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import (
    DSProposalV1ExpGen,
    DSProposalV2ExpGen,
)
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.utils.agent.tpl import T


class DSExpGen(ExpGen):
    """
    Data Science Task Generator.
    This is a experiment router generator;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen(self, trace: DSTrace) -> DSExperiment:
        # sota_exp = trace.sota_experiment()

        # # Draft
        # # TODO: draft here
        # if sota_exp is None:
        #     pass

        # Propose
        if DS_RD_SETTING.proposal_version == "v1":
            return DSProposalV1ExpGen(scen=self.scen).gen(trace=trace)
        if DS_RD_SETTING.proposal_version == "v2":
            return DSProposalV2ExpGen(scen=self.scen).gen(trace=trace)
