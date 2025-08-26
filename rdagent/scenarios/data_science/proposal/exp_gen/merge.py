"""Merge the version in different traces"""

import json
from datetime import timedelta
from typing import Dict, Tuple
from enum import Enum
from pydantic import BaseModel, Field

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.core.proposal import ExperimentFeedback, ExpGen
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.planner import DSExperimentPlan
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import DSProposalV2ExpGen
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry
from rdagent.scenarios.data_science.proposal.exp_gen.select.submit import BestValidSelector


class HypothesisComponent(str, Enum):
    DataLoadSpec = "DataLoadSpec"
    FeatureEng = "FeatureEng"
    Model = "Model"
    Ensemble = "Ensemble"
    Workflow = "Workflow"


class HypothesisSimple(BaseModel):
    hypothesis: str = Field(
        description="The statement of the hypothesis. It could be a design of a new component, or a concise, testable statement derived from previous experimental outcomes."
    )
    component: HypothesisComponent = Field(description="The component tag of the hypothesis.")


class MergeExpGen(ExpGen):
    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
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


class ExpGen2Hypothesis(DSProposalV2ExpGen):
    @wait_retry(retry_n=5)
    def hypothesis_gen(
        self,
        component_desc: str,
        sota_exp_desc: str,
        enable_idea_pool: bool,
        pipeline: bool = True,
        exp_feedback_list_desc: str = "",
        scenario_desc: str = "",
        problems: dict = {},
    ) -> Dict:
        sys_prompt = T(".merge:hypothesis_gen.system").r(
            component_desc=component_desc,
            hypothesis_output_format=T(".prompts_v2:output_format.hypothesis").r(
                pipeline=pipeline, enable_idea_pool=enable_idea_pool
            ),
            pipeline=pipeline,
        )
        user_prompt = T(".merge:hypothesis_gen.user").r(
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str | Dict[str, str | int]]],
        )
        resp_dict = json.loads(response)
        return resp_dict

    def get_exp_index(self, trace: DSTrace) -> int:
        leaves: list[int] = trace.get_leaves()
        if trace.sota_exp_to_submit is not None:
            sota_submit_value = trace.sota_exp_to_submit.result.loc["ensemble"].iloc[0]
            trace_scores = []
            for i, leaf in enumerate(leaves):
                if leaf == trace.current_selection[0]:
                    continue
                fb = trace.sota_experiment_fb(selection=(leaf,))
                if fb is None:
                    continue
                final_score = fb[0].result.loc["ensemble"].iloc[0]
                trace_scores.append((i, abs(final_score - sota_submit_value)))
            if trace_scores:
                return min(trace_scores, key=lambda item: item[1])[0]
        return next((i for i, leaf in enumerate(leaves) if leaf != trace.current_selection[0]))

    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        # Ignore the selection argument and use all leaves instead.
        sota_exp_fb = trace.sota_experiment_fb(selection=trace.current_selection)

        if sota_exp_fb:
            sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
                exp=sota_exp_fb[0],
                heading="Best previous exploration of the scenario",
            )
            eda_output = sota_exp_fb[0].experiment_workspace.file_dict.get("EDA.md", None)
        else:
            sota_exp_desc = ""
            eda_output = None

        trace_fbs: list[tuple[DSExperiment, ExperimentFeedback]] = []
        # find the best exp to merge
        leaves: list[int] = trace.get_leaves()
        max_sota_retrieved_num_per_trace = max(DS_RD_SETTING.max_sota_retrieved_num * 2 // len(leaves), 4)
        for leaf in leaves:
            if leaf == trace.current_selection[0]:
                continue

            trace_fbs.extend(
                trace.experiment_and_feedback_list_after_init(
                    return_type="sota",
                    search_type="ancestors",
                    selection=(leaf,),
                    max_retrieve_num=max_sota_retrieved_num_per_trace,
                )
            )

        success_fb_list = list(set(trace_fbs))
        logger.info(
            f"Merge Hypothesis: select {len(success_fb_list)} from {len(trace_fbs)} SOTA experiments found in {len(leaves)} traces"
        )

        if len(success_fb_list) > 0:
            exp_to_merge_fb_desc = T("scenarios.data_science.proposal.exp_gen.merge:trace").r(
                exp_and_feedback_list=success_fb_list,
                type="success",
                heading="Successful iterations:",
                success_trial_desc="These trials are the steps or changes that led to the success of the solution to be merged",
                pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
            )
        else:
            exp_index = self.get_exp_index(trace)
            exp_to_merge_fb = trace.sota_experiment_fb(selection=(exp_index,))
            if exp_to_merge_fb is None:
                exp_to_merge_fb = trace.hist[exp_index]

            exp_to_merge_fb_desc = T("scenarios.data_science.share:describe.feedback").r(
                exp_and_feedback=exp_to_merge_fb,
                heading="The feedback for the solution to be merged",
            )

        component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        hypothesis_dict = self.hypothesis_gen(
            component_desc=component_desc,
            exp_feedback_list_desc=exp_to_merge_fb_desc,
            sota_exp_desc=sota_exp_desc,
            enable_idea_pool=DS_RD_SETTING.enable_knowledge_base,
            pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
        )

        all_problems = {}
        pickled_problem_name, new_hypothesis = self.hypothesis_rank(
            hypothesis_dict=hypothesis_dict,
            problem_dict=all_problems,
            selected_idx=0,
        )
        if DS_RD_SETTING.enable_knowledge_base:
            trace.knowledge_base.update_pickled_problem(all_problems, pickled_problem_name)

        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)

        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp_fb[0] if sota_exp_fb else None,
            hypotheses=[new_hypothesis],
            pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
            failed_exp_feedback_list_desc="",
        )


class ExpGen2TraceAndMerge(ExpGen):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_exp_gen = MergeExpGen(self.scen)
        self.exp_gen = DataScienceRDLoop.default_exp_gen(self.scen)

    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        logger.info(f"Remain time: {timer.remain_time()}")

        if timer.remain_time() >= timedelta(hours=DS_RD_SETTING.merge_hours):
            leaves: list[int] = trace.get_leaves()
            if len(leaves) < 2:
                selection = trace.NEW_ROOT  # create new trace
            else:
                selection = (
                    leaves[0],
                )  # continue the first trace. This will result in the interleaving of two traces expansion.
            trace.set_current_selection(selection)
            return self.exp_gen.gen(trace)
        else:
            # disable reset in merging stage
            DS_RD_SETTING.coding_fail_reanalyze_threshold = 100000
            DS_RD_SETTING.consecutive_errors = 100000

            if trace.sub_trace_count < 2:
                return self.exp_gen.gen(trace)
            else:
                return self.merge_exp_gen.gen(trace)


class MergeExpGen_MultiTrace(ExpGen):
    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        # Ignore the selection argument and use all leaves instead.
        leaves: list[int] = trace.get_leaves()

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
        self.exp_gen = DataScienceRDLoop.default_exp_gen(self.scen)
        self.flag_start_merge = False

    def reset_exp_gen_version(self, version: str = "v2"):
        # AFAIK, this class is not used anymore (because v3 & v1 is deprecated); So we just leave a NotImplementedError instead of refine it.
        # DS_RD_SETTING.proposal_version = version
        # logger.info(f"ExpGen2TraceAndMergeV2: Resetting proposal version to {version}")
        # self.exp_gen = DataScienceRDLoop._get_exp_gen(
        #     f"rdagent.scenarios.data_science.proposal.exp_gen.DSExpGen", self.scen
        # )
        raise NotImplementedError("You should not switch version with proposal_version")

    def gen(
        self, trace: DSTrace, plan: DSExperimentPlan | None = None, selection: tuple[int, ...] = (-1,)
    ) -> DSExperiment:
        timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        logger.info(f"Remain time: {timer.remain_time()}")

        if timer.remain_time() >= timedelta(hours=DS_RD_SETTING.merge_hours):
            if DS_RD_SETTING.enable_multi_version_exp_gen:
                exp_gen_version_list = DS_RD_SETTING.exp_gen_version_list.split(",")
                for version in exp_gen_version_list:
                    assert version in ["v3", "v2", "v1"]

                if len(trace.hist) == 0:
                    # set the proposal version for the first sub-trace
                    self.reset_exp_gen_version(version=exp_gen_version_list[0])
                elif len(trace.get_current_selection()) == 0 and trace.sub_trace_count > 0:
                    # reset the proposal version at the start of other sub-trace
                    if trace.sub_trace_count - 1 < len(exp_gen_version_list):
                        self.reset_exp_gen_version(version=exp_gen_version_list[trace.sub_trace_count - 1])
                    else:
                        self.reset_exp_gen_version(version=exp_gen_version_list[-1])

            return self.exp_gen.gen(trace)

        else:
            # disable reset in merging stage
            DS_RD_SETTING.coding_fail_reanalyze_threshold = 100000
            DS_RD_SETTING.consecutive_errors = 100000

            leaves: list[int] = trace.get_leaves()
            if len(leaves) < 2:
                trace.set_current_selection(selection=(-1,))
                return self.exp_gen.gen(trace)
            else:
                if not self.flag_start_merge:  # root node of the merge trace
                    self.flag_start_merge = True
                    trace.set_current_selection(trace.NEW_ROOT)
                    return self.merge_exp_gen.gen(trace)
                else:
                    # return self.merge_exp_gen.gen(trace)
                    trace.set_current_selection(selection=(-1,))
                    return self.exp_gen.gen(trace)  # continue the last trace, to polish the merged solution


class ExpGen2TraceAndMergeV3(ExpGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_exp_gen = ExpGen2Hypothesis(self.scen)
        self.exp_gen = DataScienceRDLoop.default_exp_gen(self.scen)

    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        logger.info(f"Remain time: {timer.remain_time()}")

        if timer.remain_time() >= timedelta(hours=DS_RD_SETTING.merge_hours):
            return self.exp_gen.gen(trace)
        else:
            # disable reset in merging stage
            DS_RD_SETTING.coding_fail_reanalyze_threshold = 100000
            DS_RD_SETTING.consecutive_errors = 100000

            leaves: list[int] = trace.get_leaves()
            if len(leaves) < 2:
                trace.set_current_selection(selection=(-1,))
                return self.exp_gen.gen(trace)
            else:
                selection = (leaves[0],)
                if trace.sota_exp_to_submit is not None:
                    for i in range(1, len(leaves)):
                        if trace.is_parent(trace.exp2idx(trace.sota_exp_to_submit), leaves[i]):
                            selection = (leaves[i],)
                            break
                trace.set_current_selection(selection)
                return self.merge_exp_gen.gen(trace)

# old merge exp_gen,
class ExpGen3Hypothesis(DSProposalV2ExpGen):
    @wait_retry(retry_n=5)
    def hypothesis_gen(
        self,
        component_desc: str,
        sota_exp_desc: str,
        enable_idea_pool: bool,
        pipeline: bool = True,
        exp_feedback_list_desc: str = "",
        scenario_desc: str = "",
        problems: dict = {},
    ) -> Dict:
        sys_prompt = T(".merge:hypothesis_gen.system").r(
            component_desc=component_desc,
            hypothesis_output_format=T(".prompts_v2:output_format.hypothesis").r(
                pipeline=pipeline, enable_idea_pool=enable_idea_pool
            ),
            pipeline=pipeline,
        )
        user_prompt = T(".merge:hypothesis_gen.user").r(
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str | Dict[str, str | int]]],
        )
        resp_dict = json.loads(response)
        return resp_dict

    def hypothesis_smooth_with_llm(
        self, scenario_desc: str, exp_feedback_list_desc: str, sota_exp_desc: str, hypothesis_candidates: dict
    ):

        res_time = RD_Agent_TIMER_wrapper.timer.remain_time()
        total_time = RD_Agent_TIMER_wrapper.timer.all_duration
        use_time = round(total_time.total_seconds(), 2) - round(res_time.total_seconds(), 2)
        use_ratio = 100 * use_time / round(total_time.total_seconds(), 2)
        use_ratio = round(use_ratio, 2)

        ensemble_timeout = DS_RD_SETTING.ensemble_timeout
        hypothesis_candidates = str(json.dumps(hypothesis_candidates, indent=2))

        sys_prompt = T(".merge:hypothesis_gen_smooth.system").r(
            hypothesis_candidates=hypothesis_candidates,
            res_time=round(res_time.total_seconds(), 2),
            ensemble_timeout=ensemble_timeout,
            use_ratio=use_ratio,
            hypothesis_output_format=T(".merge:output_format.hypothesis_gen_smooth").r(
                hypothesis_candidates=hypothesis_candidates
            ),
        )
        user_prompt = T(".merge:hypothesis_gen_smooth.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format=HypothesisSimple if self.supports_response_schema else {"type": "json_object"},
            json_target_type=(
                Dict[str, Dict[str, str | Dict[str, str | int]]] if not self.supports_response_schema else None
            ),
        )

        response_dict = json.loads(response)
        return response_dict

    def get_exp_index(self, trace: DSTrace) -> int:
        leaves: list[int] = trace.get_leaves()
        if trace.sota_exp_to_submit is not None:
            sota_submit_value = trace.sota_exp_to_submit.result.loc["ensemble"].iloc[0]
            trace_scores = []
            for i, leaf in enumerate(leaves):
                if leaf == trace.current_selection[0]:
                    continue
                fb = trace.sota_experiment_fb(selection=(leaf,))
                if fb is None:
                    continue
                final_score = fb[0].result.loc["ensemble"].iloc[0]
                trace_scores.append((i, abs(final_score - sota_submit_value)))
            if trace_scores:
                return min(trace_scores, key=lambda item: item[1])[0]
        return next((i for i, leaf in enumerate(leaves) if leaf != trace.current_selection[0]))

    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        # Ignore the selection argument and use all leaves instead.
        sota_exp_fb = trace.sota_experiment_fb(selection=trace.current_selection)

        if sota_exp_fb:
            sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
                exp=sota_exp_fb[0],
                heading="Best previous exploration of the scenario",
            )
            eda_output = sota_exp_fb[0].experiment_workspace.file_dict.get("EDA.md", None)
        else:
            sota_exp_desc = ""
            eda_output = None

        trace_fbs = []
        # find the best exp to merge
        leaves: list[int] = trace.get_leaves()
        for leaf in leaves:
            if leaf == trace.current_selection[0]:
                continue

            trace_fbs.append(
                trace.experiment_and_feedback_list_after_init(
                    return_type="sota",
                    search_type="ancestors",
                    selection=(leaf,),
                )
            )

        num_to_slice = 20
        if sum(len(fb_list) for fb_list in trace_fbs) > num_to_slice:
            success_fb_trace_count = sum(1 for fb_list in trace_fbs if fb_list)
            success_fb_list = [
                fb for fb_list in trace_fbs for fb in fb_list[-(num_to_slice // success_fb_trace_count) :]
            ]
        else:
            success_fb_list = [fb for fb_list in trace_fbs for fb in fb_list]

        if len(success_fb_list) > 0:
            exp_to_merge_fb_desc = T("scenarios.data_science.proposal.exp_gen.merge:trace").r(
                exp_and_feedback_list=success_fb_list,
                type="success",
                heading="Successful iterations:",
                success_trial_desc="These trials are the steps or changes that led to the success of the solution to be merged",
                pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
            )
        else:
            exp_index = self.get_exp_index(trace)
            exp_to_merge_fb = trace.sota_experiment_fb(selection=(exp_index,))
            if exp_to_merge_fb is None:
                exp_to_merge_fb = trace.hist[exp_index]

            exp_to_merge_fb_desc = T("scenarios.data_science.share:describe.feedback").r(
                exp_and_feedback=exp_to_merge_fb,
                heading="The feedback for the solution to be merged",
            )

        component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)
        hypothesis_dict = self.hypothesis_gen(
            component_desc=component_desc,
            exp_feedback_list_desc=exp_to_merge_fb_desc,
            sota_exp_desc=sota_exp_desc,
            enable_idea_pool=DS_RD_SETTING.enable_knowledge_base,
            pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
        )

        response_dict = self.hypothesis_smooth_with_llm(
            scenario_desc=scenario_desc,
            exp_feedback_list_desc=exp_to_merge_fb_desc,
            sota_exp_desc=sota_exp_desc,
            hypothesis_candidates=hypothesis_dict,
        )
        component_map = {
            "Model": HypothesisComponent.Model,
            "Ensemble": HypothesisComponent.Ensemble,
            "Workflow": HypothesisComponent.Workflow,
            "FeatureEng": HypothesisComponent.FeatureEng,
            "DataLoadSpec": HypothesisComponent.DataLoadSpec,
        }

        comp_str = response_dict.get("component")
        hypo_str = response_dict.get("hypothesis")

        if comp_str in component_map and hypo_str is not None:
            new_hypothesis = DSHypothesis(component=component_map[comp_str], hypothesis=hypo_str)

        all_problems = None
        pickled_problem_name = None
        # all_problems = {}
        # pickled_problem_name, new_hypothesis = self.hypothesis_rank(
        #     hypothesis_dict=hypothesis_dict,
        #     problem_dict=all_problems,
        #     selected_idx=0,
        # )

        if DS_RD_SETTING.enable_knowledge_base:
            trace.knowledge_base.update_pickled_problem(all_problems, pickled_problem_name)

        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp_fb[0] if sota_exp_fb else None,
            hypotheses=[new_hypothesis],
            pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
            failed_exp_feedback_list_desc="",
        )


# new merge+ smooth + max ,
class ExpGen4Hypothesis(DSProposalV2ExpGen):
    @wait_retry(retry_n=5)
    def hypothesis_gen(
        self,
        component_desc: str,
        sota_exp_desc: str,
        enable_idea_pool: bool,
        pipeline: bool = True,
        exp_feedback_list_desc: str = "",
        scenario_desc: str = "",
        problems: dict = {},
    ) -> Dict:
        sys_prompt = T(".merge:hypothesis_gen.system").r(
            component_desc=component_desc,
            hypothesis_output_format=T(".prompts_v2:output_format.hypothesis").r(
                pipeline=pipeline, enable_idea_pool=enable_idea_pool
            ),
            pipeline=pipeline,
        )
        user_prompt = T(".merge:hypothesis_gen.user").r(
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str | Dict[str, str | int]]],
        )
        resp_dict = json.loads(response)
        return resp_dict

    def get_exp_index(self, trace: DSTrace) -> int:
        leaves: list[int] = trace.get_leaves()
        bvs = BestValidSelector()
        sota_exp = bvs.get_sota_exp_to_submit(trace)
        sota_flag = (sota_exp is not None and sota_exp.result is not None)
        if sota_flag:
            sota_submit_value =sota_exp.result.loc["ensemble"].iloc[0]
            trace_scores = []
            for i, leaf in enumerate(leaves):
                if leaf == trace.current_selection[0]:
                    continue
                fb = trace.sota_experiment_fb(selection=(leaf,))
                if fb is None:
                    continue
                final_score = fb[0].result.loc["ensemble"].iloc[0]
                trace_scores.append((i, abs(final_score - sota_submit_value)))
            if trace_scores:
                return min(trace_scores, key=lambda item: item[1])[0]
        return next((i for i, leaf in enumerate(leaves) if leaf != trace.current_selection[0]))
    
    def hypothesis_smooth_with_llm(
        self, exp_feedback_list_desc: str, sota_exp_desc: str, hypothesis_candidates: dict
    ):

        res_time = RD_Agent_TIMER_wrapper.timer.remain_time()
        total_time = RD_Agent_TIMER_wrapper.timer.all_duration
        use_time = round(total_time.total_seconds(), 2) - round(res_time.total_seconds(), 2)
        use_ratio = 100 * use_time / round(total_time.total_seconds(), 2)
        use_ratio = round(use_ratio, 2)

        ensemble_timeout = DS_RD_SETTING.ensemble_timeout
        hypothesis_candidates = str(json.dumps(hypothesis_candidates, indent=2))

        sys_prompt = T(".merge:hypothesis_gen_smooth.system").r(
            hypothesis_candidates=hypothesis_candidates,
            res_time=round(res_time.total_seconds(), 2),
            ensemble_timeout=ensemble_timeout,
            use_ratio=use_ratio,
            hypothesis_output_format=T(".merge:output_format.hypothesis_gen_smooth").r(
                hypothesis_candidates=hypothesis_candidates
            ),
        )
        user_prompt = T(".merge:hypothesis_gen_smooth.user").r(
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format=HypothesisSimple if self.supports_response_schema else {"type": "json_object"},
            json_target_type=(
                Dict[str, Dict[str, str | Dict[str, str | int]]] if not self.supports_response_schema else None
            ),
        )

        response_dict = json.loads(response)
        return response_dict
    
    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        # Ignore the selection argument and use all leaves instead.
        sota_exp_fb = trace.sota_experiment_fb(selection=trace.current_selection)

        if sota_exp_fb:
            sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
                exp=sota_exp_fb[0],
                heading="Best previous exploration of the scenario",
            )
            eda_output = sota_exp_fb[0].experiment_workspace.file_dict.get("EDA.md", None)
        else:
            sota_exp_desc = ""
            eda_output = None

        trace_fbs: list[tuple[DSExperiment, ExperimentFeedback]] = []
        # find the best exp to merge
        leaves: list[int] = trace.get_leaves()
        max_sota_retrieved_num_per_trace = max(DS_RD_SETTING.max_sota_retrieved_num * 2 // len(leaves), 4)
        for leaf in leaves:
            if leaf == trace.current_selection[0]:
                continue

            trace_fbs.extend(
                trace.experiment_and_feedback_list_after_init(
                    return_type="sota",
                    search_type="ancestors",
                    selection=(leaf,),
                    max_retrieve_num=max_sota_retrieved_num_per_trace,
                )
            )

        success_fb_list = list(set(trace_fbs))
        logger.info(
            f"Merge Hypothesis: select {len(success_fb_list)} from {len(trace_fbs)} SOTA experiments found in {len(leaves)} traces"
        )

        if len(success_fb_list) > 0:
            exp_to_merge_fb_desc = T("scenarios.data_science.proposal.exp_gen.merge:trace").r(
                exp_and_feedback_list=success_fb_list,
                type="success",
                heading="Successful iterations:",
                success_trial_desc="These trials are the steps or changes that led to the success of the solution to be merged",
                pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
            )
        else:
            exp_index = self.get_exp_index(trace)
            exp_to_merge_fb = trace.sota_experiment_fb(selection=(exp_index,))
            if exp_to_merge_fb is None:
                exp_to_merge_fb = trace.hist[exp_index]

            exp_to_merge_fb_desc = T("scenarios.data_science.share:describe.feedback").r(
                exp_and_feedback=exp_to_merge_fb,
                heading="The feedback for the solution to be merged",
            )

        component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        hypothesis_dict = self.hypothesis_gen(
            component_desc=component_desc,
            exp_feedback_list_desc=exp_to_merge_fb_desc,
            sota_exp_desc=sota_exp_desc,
            enable_idea_pool=DS_RD_SETTING.enable_knowledge_base,
            pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
        )

        # all_problems = {}
        # pickled_problem_name, new_hypothesis = self.hypothesis_rank(
        #     hypothesis_dict=hypothesis_dict,
        #     problem_dict=all_problems,
        #     selected_idx=0,
        # )
        response_dict = self.hypothesis_smooth_with_llm(
            exp_feedback_list_desc=exp_to_merge_fb_desc,
            sota_exp_desc=sota_exp_desc,
            hypothesis_candidates=hypothesis_dict,
        )
        component_map = {
            "Model": HypothesisComponent.Model,
            "Ensemble": HypothesisComponent.Ensemble,
            "Workflow": HypothesisComponent.Workflow,
            "FeatureEng": HypothesisComponent.FeatureEng,
            "DataLoadSpec": HypothesisComponent.DataLoadSpec,
        }

        comp_str = response_dict.get("component")
        hypo_str = response_dict.get("hypothesis")

        if comp_str in component_map and hypo_str is not None:
            new_hypothesis = DSHypothesis(component=component_map[comp_str], hypothesis=hypo_str)

        all_problems = None
        pickled_problem_name = None
        
        if DS_RD_SETTING.enable_knowledge_base:
            trace.knowledge_base.update_pickled_problem(all_problems, pickled_problem_name)

        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)

        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp_fb[0] if sota_exp_fb else None,
            hypotheses=[new_hypothesis],
            pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
            failed_exp_feedback_list_desc="",
        )
    


# 5 compare with ExpGen2Hypothesis, max
class ExpGen5Hypothesis(DSProposalV2ExpGen):
    @wait_retry(retry_n=5)
    def hypothesis_gen(
        self,
        component_desc: str,
        sota_exp_desc: str,
        enable_idea_pool: bool,
        pipeline: bool = True,
        exp_feedback_list_desc: str = "",
        scenario_desc: str = "",
        problems: dict = {},
    ) -> Dict:
        sys_prompt = T(".merge:hypothesis_gen.system").r(
            component_desc=component_desc,
            hypothesis_output_format=T(".prompts_v2:output_format.hypothesis").r(
                pipeline=pipeline, enable_idea_pool=enable_idea_pool
            ),
            pipeline=pipeline,
        )
        user_prompt = T(".merge:hypothesis_gen.user").r(
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str | Dict[str, str | int]]],
        )
        resp_dict = json.loads(response)
        return resp_dict

    def get_exp_index(self, trace: DSTrace) -> int:
        leaves: list[int] = trace.get_leaves()
        bvs = BestValidSelector()
        sota_exp = bvs.get_sota_exp_to_submit(trace)
        sota_flag = (sota_exp is not None and sota_exp.result is not None)
        if sota_flag:
            sota_submit_value =sota_exp.result.loc["ensemble"].iloc[0]
            trace_scores = []
            for i, leaf in enumerate(leaves):
                if leaf == trace.current_selection[0]:
                    continue
                fb = trace.sota_experiment_fb(selection=(leaf,))
                if fb is None:
                    continue
                final_score = fb[0].result.loc["ensemble"].iloc[0]
                trace_scores.append((i, abs(final_score - sota_submit_value)))
            if trace_scores:
                return min(trace_scores, key=lambda item: item[1])[0]
        return next((i for i, leaf in enumerate(leaves) if leaf != trace.current_selection[0]))

    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        # Ignore the selection argument and use all leaves instead.
        sota_exp_fb = trace.sota_experiment_fb(selection=trace.current_selection)

        if sota_exp_fb:
            sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
                exp=sota_exp_fb[0],
                heading="Best previous exploration of the scenario",
            )
            eda_output = sota_exp_fb[0].experiment_workspace.file_dict.get("EDA.md", None)
        else:
            sota_exp_desc = ""
            eda_output = None

        trace_fbs: list[tuple[DSExperiment, ExperimentFeedback]] = []
        # find the best exp to merge
        leaves: list[int] = trace.get_leaves()
        max_sota_retrieved_num_per_trace = max(DS_RD_SETTING.max_sota_retrieved_num * 2 // len(leaves), 4)
        for leaf in leaves:
            if leaf == trace.current_selection[0]:
                continue

            trace_fbs.extend(
                trace.experiment_and_feedback_list_after_init(
                    return_type="sota",
                    search_type="ancestors",
                    selection=(leaf,),
                    max_retrieve_num=max_sota_retrieved_num_per_trace,
                )
            )

        success_fb_list = list(set(trace_fbs))
        logger.info(
            f"Merge Hypothesis: select {len(success_fb_list)} from {len(trace_fbs)} SOTA experiments found in {len(leaves)} traces"
        )

        if len(success_fb_list) > 0:
            exp_to_merge_fb_desc = T("scenarios.data_science.proposal.exp_gen.merge:trace").r(
                exp_and_feedback_list=success_fb_list,
                type="success",
                heading="Successful iterations:",
                success_trial_desc="These trials are the steps or changes that led to the success of the solution to be merged",
                pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
            )
        else:
            exp_index = self.get_exp_index(trace)
            exp_to_merge_fb = trace.sota_experiment_fb(selection=(exp_index,))
            if exp_to_merge_fb is None:
                exp_to_merge_fb = trace.hist[exp_index]

            exp_to_merge_fb_desc = T("scenarios.data_science.share:describe.feedback").r(
                exp_and_feedback=exp_to_merge_fb,
                heading="The feedback for the solution to be merged",
            )

        component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        hypothesis_dict = self.hypothesis_gen(
            component_desc=component_desc,
            exp_feedback_list_desc=exp_to_merge_fb_desc,
            sota_exp_desc=sota_exp_desc,
            enable_idea_pool=DS_RD_SETTING.enable_knowledge_base,
            pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
        )

        all_problems = {}
        pickled_problem_name, new_hypothesis = self.hypothesis_rank(
            hypothesis_dict=hypothesis_dict,
            problem_dict=all_problems,
            selected_idx=0,
        )
        if DS_RD_SETTING.enable_knowledge_base:
            trace.knowledge_base.update_pickled_problem(all_problems, pickled_problem_name)

        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)

        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp_fb[0] if sota_exp_fb else None,
            hypotheses=[new_hypothesis],
            pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
            failed_exp_feedback_list_desc="",
        )
