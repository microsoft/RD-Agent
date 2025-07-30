from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExpGen
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import DSProposalV2ExpGen
from rdagent.utils.agent.tpl import T


class FinetuneExpGen(DSProposalV2ExpGen):
    def gen(
        self,
        trace: DSTrace,
    ) -> DSExperiment:
        component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()

        if (sota_exp_fb := trace.sota_experiment_fb()) is None:
            sota_exp, fb_to_sota_exp = None, None
        else:
            sota_exp, fb_to_sota_exp = sota_exp_fb

        if not isinstance(sota_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = sota_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = self.scen.get_scenario_all_desc(eda_output=eda_output)

        # TODO: this is a over simplified version. More features will be added after more survey
        sota_exp_desc = "No previous SOTA experiments available."
        failed_exp_feedback_list_desc = "No previous experiments available."

        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp,
            hypotheses=[
                DSHypothesis(
                    component="Model",
                )
            ],
            pipeline=True,
            failed_exp_feedback_list_desc=failed_exp_feedback_list_desc,
            fb_to_sota_exp=fb_to_sota_exp,
        )
