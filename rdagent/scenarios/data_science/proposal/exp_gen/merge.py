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
        sota_exp = trace.sota_experiment(selection=(leaves[0],))
        exp_to_merge = trace.sota_experiment(selection=(leaves[1],))

        # scenario_desc = trace.scen.get_scenario_all_desc()
        # scenario_desc is not needed in task description. So we have to do it.

        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp,
            heading="Best of previous exploration of the scenario",
        )
        exp_to_merge_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=exp_to_merge,
            heading="A solution that to be merged into previous best solution",
        )

        exp_and_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(
                return_type="sota", selection=(leaves[1],)
            ),
            type="success",
        )

        task = PipelineTask(
            description=T("scenarios.data_science.proposal.exp_gen.merge:task").r(
                sota_exp_desc=sota_exp_desc,
                exp_to_merge_desc=exp_to_merge_desc,
                exp_and_feedback_list_desc=exp_and_feedback_list_desc,
            )
        )

        exp = DSExperiment(
            pending_tasks_list=[[task]],
            hypothesis=DSHypothesis(
                component="Pipeline",
                hypothesis="Merging two different versions of solutions would get the best of both sides and result in a better solution",
            ),
        )

        if sota_exp is not None:
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)
        return exp
