"""Merge the version in different traces"""

from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.core.proposal import ExpGen
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.utils.agent.tpl import T


class MergeExpGen(ExpGen):
    def gen(self, trace: DSTrace, selection: tuple[int, ...] = (-1,)) -> DSExperiment:
        # Ignore the selection argument and use all leaves instead.
        leaves: list[int] = trace.get_leaves()
        trace.set_current_selection((leaves[0],))  # override the current selection.

        # assuming merging the first and sencond trace.
        sota_exp_fb = trace.sota_experiment_fb(selection=(leaves[0],))
        if sota_exp_fb is None:
            sota_exp_fb = trace.hist[leaves[0]]
        exp_to_merge_fb = trace.sota_experiment_fb(selection=(leaves[1],))
        if exp_to_merge_fb is None:
            exp_to_merge_fb = trace.hist[leaves[1]]

        # scenario_desc = trace.scen.get_scenario_all_desc()
        # scenario_desc is not needed in task description. So we have to do it.

        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp_fb[0],
            heading="Best previous exploration of the scenario",
        )
        sota_exp_fb_desc = T("scenarios.data_science.share:describe.feedback").r(
            exp_and_feedback=sota_exp_fb,
            heading="The feedback for best previous exploration",
        )
        exp_to_merge_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=exp_to_merge_fb[0],
            heading="A solution that to be merged into previous best solution",
        )

        success_fb_list = trace.experiment_and_feedback_list_after_init(return_type="sota", selection=(leaves[1],))
        if len(success_fb_list) > 0:
            exp_to_merge_fb_desc = T("scenarios.data_science.share:describe.trace").r(
                exp_and_feedback_list=success_fb_list, type="success", heading="Successful iterations:"
            )
        else:
            exp_to_merge_fb_desc = T("scenarios.data_science.share:describe.feedback").r(
                exp_and_feedback=exp_to_merge_fb,
                heading="The feedback for the solution to be merged",
            )

        task = PipelineTask(
            description=T("scenarios.data_science.proposal.exp_gen.merge:task").r(
                sota_exp_desc=sota_exp_desc,
                sota_exp_fb_desc=sota_exp_fb_desc,
                exp_to_merge_desc=exp_to_merge_desc,
                exp_to_merge_fb_desc=exp_to_merge_fb_desc,
            )
        )

        exp = DSExperiment(
            pending_tasks_list=[[task]],
            hypothesis=DSHypothesis(
                component="Pipeline",
                hypothesis="Merging two different versions of solutions would get the best of both sides and result in a better solution",
            ),
        )

        if sota_exp_fb is not None:
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp_fb[0].experiment_workspace)
        return exp
