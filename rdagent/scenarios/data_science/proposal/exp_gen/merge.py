"""Merge the version in different traces"""

from datetime import timedelta

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.core.proposal import ExpGen
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSExpGen
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

        success_fb_list = trace.experiment_and_feedback_list_after_init(
            return_type="sota", search_type="ancestors", selection=(leaves[1],)
        )
        if len(success_fb_list) > 0:
            exp_to_merge_fb_desc = T("scenarios.data_science.share:describe.trace").r(
                exp_and_feedback_list=success_fb_list,
                type="success",
                heading="Successful iterations:",
                success_trial_desc="These trials are the steps or changes that led to the success of the solution to be merged",
                pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
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


# dual-target version
class ExpGen2TraceAndMerge(ExpGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_exp_gen = MergeExpGen(self.scen)
        self.exp_gen = DSExpGen(self.scen)

    def gen(self, trace: DSTrace, selection: tuple[int, ...] = (-1,)) -> DSExperiment:
        timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        logger.info(f"Remain time: {timer.remain_time_duration}")

        if timer.remain_time_duration >= timedelta(hours=DS_RD_SETTING.merge_hours):
            leaves: list[int] = trace.get_leaves()
            if len(leaves) < 2:
                selection = tuple()  # create new trace
            else:
                selection = (
                    leaves[0],
                )  # continue the first trace. This will result in the interleaving of two traces expansion.
            return self.exp_gen.gen(trace, selection)
        else:
            # disable reset in merging stage
            DS_RD_SETTING.coding_fail_reanalyze_threshold = 100000
            DS_RD_SETTING.consecutive_errors = 100000

            leaves: list[int] = trace.get_leaves()
            if len(leaves) < 2:
                return self.exp_gen.gen(trace, selection)
            else:
                return self.merge_exp_gen.gen(trace, selection)


class MergeExpGen_MultiTrace(ExpGen):
    def gen(self, trace: DSTrace, selection: tuple[int, ...] = (-1,)) -> DSExperiment:
        # Ignore the selection argument and use all leaves instead.
        leaves: list[int] = trace.get_leaves()
        trace.set_current_selection(selection)  #

        # assuming merging the first and sencond trace.
        sota_exp_fb = trace.sota_experiment_fb(selection=(leaves[0],))
        if sota_exp_fb is None:
            sota_exp_fb = trace.hist[leaves[0]]

        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp_fb[0],
            heading="Best previous exploration of the scenario",
        )
        sota_exp_fb_desc = T("scenarios.data_science.share:describe.feedback").r(
            exp_and_feedback=sota_exp_fb,
            heading="The feedback for best previous exploration",
        )

        exp_fb_desc_to_merge_list = []
        # find the best exp to merge
        for i in range(1, len(leaves)):
            exp_to_merge_fb = trace.sota_experiment_fb(selection=(leaves[i],))
            if exp_to_merge_fb is None:
                exp_to_merge_fb = trace.hist[leaves[i]]

            exp_to_merge_desc = T("scenarios.data_science.share:describe.exp").r(
                exp=exp_to_merge_fb[0],
                heading="A solution that to be merged into previous best solution",
            )

            success_fb_list = trace.experiment_and_feedback_list_after_init(
                return_type="sota",
                search_type="ancestors",
                selection=(leaves[i],),
            )
            if len(success_fb_list) > 0:

                exp_to_merge_fb_desc = T("scenarios.data_science.share:describe.trace").r(
                    exp_and_feedback_list=success_fb_list,
                    type="success",
                    heading="Successful iterations:",
                    success_trial_desc="These trials are the steps or changes that led to the success of the solution to be merged",
                    pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
                )
            else:
                exp_to_merge_fb_desc = T("scenarios.data_science.share:describe.feedback").r(
                    exp_and_feedback=exp_to_merge_fb,
                    heading="The feedback for the solution to be merged",
                )

        exp_fb_desc_to_merge_list.append((exp_to_merge_desc, exp_to_merge_fb_desc))

        task = PipelineTask(
            description=T("scenarios.data_science.proposal.exp_gen.merge:multi_trace").r(
                sota_exp_desc=sota_exp_desc,
                sota_exp_fb_desc=sota_exp_fb_desc,
                exp_fb_desc_to_merge_list=exp_fb_desc_to_merge_list,
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


# multi-target version
# allow multiple traces to grow and then merge
class ExpGen2TraceAndMergeV2(ExpGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_exp_gen = MergeExpGen_MultiTrace(self.scen)
        self.exp_gen = DSExpGen(self.scen)
        self.MAX_TRACE_NUM = DS_RD_SETTING.max_trace_num  # maximum number of traces to grow before merging
        self.flag_start_merge = False

    def reset_exp_gen_version(self, version: str = "v2"):
        DS_RD_SETTING.proposal_version = version
        logger.info(f"ExpGen2TraceAndMergeV2: Resetting proposal version to {version}")

    def gen(self, trace: DSTrace, selection: tuple[int, ...] = (-1,)) -> DSExperiment:
        timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        logger.info(f"Remain time: {timer.remain_time_duration}")

        if timer.remain_time_duration >= timedelta(hours=DS_RD_SETTING.merge_hours):

            if DS_RD_SETTING.enable_inject_knowledge_at_root:
                if DS_RD_SETTING.knowledge_base_path is not None and DS_RD_SETTING.idea_pool_json_path is not None:
                    if len(trace.hist) == 0:
                        # set the knowledge base option to True for the first trace
                        DS_RD_SETTING.enable_knowledge_base = True

            if DS_RD_SETTING.enable_multi_version_exp_gen:
                exp_gen_version_list = DS_RD_SETTING.exp_gen_version_list.split(",")
                for version in exp_gen_version_list:
                    assert version in ["v3", "v2", "v1"]

                if len(trace.hist) == 0:
                    # set the proposal version for the first sub-trace
                    self.reset_exp_gen_version(version=exp_gen_version_list[0])
                elif len(trace.get_current_selection()) == 0 and trace.get_sub_trace_count() > 0:
                    # reset the proposal version at the start of other sub-trace
                    if trace.get_sub_trace_count() - 1 < len(exp_gen_version_list):
                        self.reset_exp_gen_version(version=exp_gen_version_list[trace.get_sub_trace_count() - 1])
                    else:
                        self.reset_exp_gen_version(version=exp_gen_version_list[-1])

            return self.exp_gen.gen(trace, selection)

        else:
            # disable reset in merging stage
            DS_RD_SETTING.coding_fail_reanalyze_threshold = 100000
            DS_RD_SETTING.consecutive_errors = 100000

            leaves: list[int] = trace.get_leaves()
            if len(leaves) < 2:
                return self.exp_gen.gen(trace, selection=(-1,))
            else:

                if not self.flag_start_merge:  # root node of the merge trace
                    self.flag_start_merge = True
                    selection = tuple()
                    return self.merge_exp_gen.gen(trace, selection)
                else:
                    # return self.merge_exp_gen.gen(trace, selection)
                    return self.exp_gen.gen(
                        trace, selection=(-1,)
                    )  # continue the last trace, to polish the merged solution
