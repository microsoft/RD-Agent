"""
The most naive way to design experiments
"""
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.core.proposal import ExpGen
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry


class NaiveExpGen(ExpGen):
    def gen(self, trace: DSTrace) -> DSExperiment:
        sota_exp = trace.sota_experiment()
        scenario_desc = trace.scen.get_scenario_all_desc()
        competition_desc = trace.scen.get_competition_full_desc()
        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="Best of previous exploration of the scenario"
        )

        sota_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="sota")
        failed_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="failed")[-DS_RD_SETTING.max_trace_hist:]

        sota_exp_and_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=sota_exp_feedback_list,
            success=True,
        )
        failed_exp_and_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=failed_exp_feedback_list,
            success=False,
        )

        sys_prompt = T(".naive:naive_gen.system").r()

        user_prompt = T(".naive:naive_gen.user").r(
            competition_desc=competition_desc,
            sota_exp_desc=sota_exp_desc,
            scenario_desc=scenario_desc,
            sota_exp_and_feedback_list_desc=sota_exp_and_feedback_list_desc,
            failed_exp_and_feedback_list_desc=failed_exp_and_feedback_list_desc,
        )

        task = build_cls_from_json_with_retry(
            cls=PipelineTask,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            retry_n=5,
        )

        exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=DSHypothesis(
            component="Pipeline",
            hypothesis=task.description,
        ))

        if sota_exp is not None:
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)
        return exp
