import json
from typing import Dict, Tuple

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.proposal import ExpGen
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.idea_pool import DSIdea
from rdagent.utils.agent.tpl import T
from rdagent.utils.repo.diff import generate_diff_from_dict
from rdagent.utils.workflow import wait_retry

COMPONENT_TASK_MAPPING = {
    "DataLoadSpec": {
        "target_name": "Data loader and specification generation",
        "spec_file": "spec/data_loader.md",
        "task_output_format": T(".prompts:output_format.data_loader").r(),
        "task_class": DataLoaderTask,
    },
    "FeatureEng": {
        "target_name": "Feature engineering",
        "spec_file": "spec/feature.md",
        "task_output_format": T(".prompts:output_format.feature").r(),
        "task_class": FeatureTask,
    },
    "Model": {
        "target_name": "Model",
        "spec_file": "spec/model.md",
        "task_output_format": T(".prompts:output_format.model").r(),
        "task_class": ModelTask,
    },
    "Ensemble": {
        "target_name": "Ensemble",
        "spec_file": "spec/ensemble.md",
        "task_output_format": T(".prompts:output_format.ensemble").r(),
        "task_class": EnsembleTask,
    },
    "Workflow": {
        "target_name": "Workflow",
        "spec_file": "spec/workflow.md",
        "task_output_format": T(".prompts:output_format.workflow").r(),
        "task_class": WorkflowTask,
    },
    "Pipeline": {
        "target_name": "Pipeline",
        "task_output_format": T(".prompts:output_format.pipeline").r(),
        "task_class": PipelineTask,
    },
}


class DSProposalV1ExpGen(ExpGen):
    def gen(self, trace: DSTrace) -> DSExperiment:
        # Guidelines:
        # System prompts: Shared condition you are facing
        # - scenario description: `scenario_desc`
        # - expected output format
        # User prompts: Task Specific information
        # - Previous Feedback
        # - Current sota implementation (encourage change based on it)
        # - Extra RAG
        sota_exp = trace.sota_experiment()
        if not isinstance(sota_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = sota_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)

        assert sota_exp is not None, "SOTA experiment is not provided."
        last_exp = trace.last_exp()
        # exp_and_feedback = trace.hist[-1]
        # last_exp = exp_and_feedback[0]

        # Step 1: Generate component
        # Describe current best solution using shared template
        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="Best of previous exploration of the scenario"
        )
        last_exp_diff = "\n".join(
            generate_diff_from_dict(sota_exp.experiment_workspace.file_dict, last_exp.experiment_workspace.file_dict)
        )  # we use file_dict for hitting the cache when replicate the experiment in another machine.

        all_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="all")

        exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=all_exp_feedback_list,
            type="all",
        )

        # Generate component using template with proper context
        component_sys_prompt = T(".prompts:component_gen.system").r(
            scenario=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            last_exp_diff=last_exp_diff,
            component_desc="\n".join(
                [
                    f"[{key}] {value}"
                    for key, value in T("scenarios.data_science.share:component_description").template.items()
                ]
            ),
        )

        component_user_prompt = T(".prompts:component_gen.user").r(
            exp_and_feedback_list_desc=exp_feedback_list_desc,
        )

        resp_dict_component: dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                component_user_prompt, component_sys_prompt, json_mode=True, json_target_type=Dict[str, str]
            )
        )

        component = resp_dict_component.get("component", "Component not provided")
        component_reason = resp_dict_component.get("reason", "Reason not provided")
        sota_exp_model_file_count = len(
            [
                k
                for k in sota_exp.experiment_workspace.file_dict.keys()
                if k.endswith(".py") and "test" not in k and k.startswith("model")
            ]
        )
        if sota_exp_model_file_count <= 1 and component == "Ensemble":
            component = "Model"

        # Why we should split component selection and steps after?
        # - after we know the selected component, we can use RAG.

        # Step 2: Generate the rest of the hypothesis & task
        component_info = COMPONENT_TASK_MAPPING.get(component)

        if component_info:
            if DS_RD_SETTING.spec_enabled:
                task_spec = sota_exp.experiment_workspace.file_dict[component_info["spec_file"]]
            else:
                task_spec = T(f"scenarios.data_science.share:component_spec.{component}").r()
            system_prompt = T(".prompts:direct_exp_gen.system").r(
                targets=component_info["target_name"],
                component=component,
                scenario=scenario_desc,
                hypothesis_specification=T(".prompts:hypothesis_specification").r(),
                hypothesis_output_format=T(".prompts:output_format.hypothesis").r(),
                task_specification=task_spec,
                task_output_format=component_info["task_output_format"],
                workflow_check=(not component == "Workflow"),
            )

            user_prompt = T(".prompts:direct_exp_gen.user").r(
                targets=component_info["target_name"],
                sota_exp_desc=sota_exp_desc,
                exp_and_feedback_list_desc=exp_feedback_list_desc,
                last_exp_diff=last_exp_diff,
            )

            def _append_retry(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
                # Only modify the user_prompt on retries (i > 0)
                user_prompt = args[0]
                user_prompt += "\n\nretrying..."
                return (user_prompt,), kwargs

            @wait_retry(retry_n=5, transform_args_fn=_append_retry)
            def _f(user_prompt):
                resp_dict = json.loads(
                    APIBackend().build_messages_and_create_chat_completion(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_mode=True,
                        # NOTE: corner cases.
                        # workflow_update may be a string
                        # model could have 2 level nested dict.
                        json_target_type=dict[str, dict[str, str | dict] | str],
                    )
                )
                assert "hypothesis_proposal" in resp_dict, "Hypothesis proposal not provided."
                assert "task_design" in resp_dict, "Task design not provided."
                task_class = component_info["task_class"]
                hypothesis_proposal = resp_dict.get("hypothesis_proposal", {})
                hypothesis = DSHypothesis(
                    component=component,
                    hypothesis=hypothesis_proposal.get("hypothesis", ""),
                    reason=component_reason + "\n" + hypothesis_proposal.get("reason", ""),
                    concise_reason=hypothesis_proposal.get("concise_reason", ""),
                    concise_observation=hypothesis_proposal.get("concise_observation", ""),
                    concise_justification=hypothesis_proposal.get("concise_justification", ""),
                    concise_knowledge=hypothesis_proposal.get("concise_knowledge", ""),
                )

                task_design = resp_dict.get("task_design", {})
                task_name = task_design["model_name"] if component == "Model" else component
                description = task_design.get(
                    "description", f"{component_info['target_name']} description not provided"
                )
                task = task_class(
                    name=task_name,
                    description=description,
                    **{k: task_design.get(k, v) for k, v in component_info.get("extra_params", {}).items()},
                )
                new_workflow_desc = resp_dict.get("workflow_update", "No update needed")
                return hypothesis, task, new_workflow_desc

            hypothesis, task, new_workflow_desc = _f(user_prompt)

            exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)
            # exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)

            if new_workflow_desc != "No update needed":
                workflow_task = WorkflowTask(
                    name="Workflow",
                    description=new_workflow_desc,
                )
                exp.pending_tasks_list.append([workflow_task])
            return exp
        else:
            raise ValueError(f"Unknown component: {component}")


class DSProposalV2ExpGen(ExpGen):
    def identify_scenario_problem(self, scenario_desc: str, sota_exp_desc: str) -> Dict:
        sys_prompt = T(".prompts_v2:scenario_problem.system").r(
            problem_spec=T(".prompts_v2:specification.problem").r(),
            problem_output_format=T(".prompts_v2:output_format.problem").r(),
        )
        user_prompt = T(".prompts_v2:scenario_problem.user").r(
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str]],
        )
        return json.loads(response)

    def identify_feedback_problem(self, scenario_desc: str, exp_feedback_list_desc: str, sota_exp_desc: str) -> Dict:
        sys_prompt = T(".prompts_v2:feedback_problem.system").r(
            problem_spec=T(".prompts_v2:specification.problem").r(),
            problem_output_format=T(".prompts_v2:output_format.problem").r(),
        )
        user_prompt = T(".prompts_v2:feedback_problem.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str]],
        )
        return json.loads(response)

    @wait_retry(retry_n=5)
    def hypothesis_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        exp_feedback_list_desc: str,
        sota_exp_desc: str,
        problems: dict,
        pipeline: bool,
        enable_idea_pool: bool,
    ) -> Dict:
        problem_formatted_str = ""
        for problem_name, problem_dict in problems.items():
            problem_formatted_str += f"# Problem Name: {problem_name}\n"
            problem_formatted_str += f"- Problem Description: {problem_dict['problem']}\n"
            if "idea" in problem_dict:
                idea_formatted_str = DSIdea(problem_dict["idea"]).to_formatted_str()
                problem_formatted_str += f"- Sampled Idea by user: \n{idea_formatted_str}\n"
            problem_formatted_str += "\n\n"

        sys_prompt = T(".prompts_v2:hypothesis_gen.system").r(
            component_desc=component_desc,
            hypothesis_spec=T(".prompts_v2:specification.hypothesis").r(pipeline=pipeline),
            hypothesis_output_format=T(".prompts_v2:output_format.hypothesis").r(
                pipeline=pipeline, enable_idea_pool=enable_idea_pool
            ),
            pipeline=pipeline,
            enable_idea_pool=enable_idea_pool,
        )
        user_prompt = T(".prompts_v2:hypothesis_gen.user").r(
            scenario_desc=scenario_desc,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            problems=problem_formatted_str,
            enable_idea_pool=enable_idea_pool,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str | Dict[str, str | int]]],
        )
        resp_dict = json.loads(response)
        return resp_dict

    def hypothesis_rank(
        self,
        hypothesis_dict: dict,
        problem_dict: dict,
        trace: DSTrace,
    ) -> Tuple[str, DSHypothesis]:
        weights = {
            "alignment_score": 0.2,
            "impact_score": 0.4,
            "novelty_score": 0.2,
            "feasibility_score": 0.1,
            "risk_reward_balance_score": 0.1,
        }
        scores_dict = {}
        for problem_name in hypothesis_dict:
            if "hypothesis" not in hypothesis_dict[problem_name]:
                continue
            scores_dict[problem_name] = {}
            for score_key in weights:
                if score_key not in hypothesis_dict[problem_name]["evaluation"]:
                    scores_dict[problem_name][score_key] = 0
                else:
                    try:
                        scores_dict[problem_name][score_key] = (
                            float(hypothesis_dict[problem_name]["evaluation"][score_key]) * weights[score_key]
                        )
                    except (ValueError, TypeError):
                        scores_dict[problem_name][score_key] = 0

        scores = pd.DataFrame(scores_dict)
        scores_sorted = scores.sum().sort_values(ascending=False)
        scores_sorted = scores_sorted[:5]  # Select top 5 hypotheses

        # Increase the weight of the hypothesis that is inspired by the idea pool to 3x.
        # Linear decay the weight of the scenario problem from 3x to 1x.
        index_to_pick_pool_list = []
        for j, problem_name in enumerate(scores_sorted.index):
            if hypothesis_dict[problem_name].get("inspired", False):
                index_to_pick_pool_list.extend([j] * 4)
            elif problem_dict.get(problem_name, {}).get("label", "") == "SCENARIO_PROBLEM":
                index_to_pick_pool_list.extend([j] * (3 - len(trace.hist) // 3))
            else:
                index_to_pick_pool_list.extend([j] * 2)
        logger.info(f"index_to_pick_pool_list: {index_to_pick_pool_list}")

        # Create a random but reproducible integer
        reproducible_int = int.from_bytes(bytes.fromhex(md5_hash(scores_sorted.to_string())), byteorder="big") % len(
            index_to_pick_pool_list
        )
        selected_idx = index_to_pick_pool_list[reproducible_int]
        max_score_problem_name = scores_sorted.index[selected_idx]
        problem_dict = problem_dict.get(max_score_problem_name, {})

        return max_score_problem_name, DSHypothesis(
            component=hypothesis_dict[max_score_problem_name].get("component", "Model"),
            hypothesis=hypothesis_dict[max_score_problem_name].get("hypothesis", "Hypothesis not provided"),
            reason=hypothesis_dict[max_score_problem_name].get("reason", "Reason not provided"),
            problem_name=max_score_problem_name,
            problem_desc=problem_dict.get("problem", "Problem description not provided"),
            problem_label=problem_dict.get("label", "FEEDBACK_PROBLEM"),
        )

    def task_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        sota_exp_desc: str,
        sota_exp: DSExperiment,
        hypothesis: DSHypothesis,
        pipeline: bool,
        failed_exp_feedback_list_desc: str,
    ) -> DSExperiment:
        if pipeline:
            component_info = COMPONENT_TASK_MAPPING["Pipeline"]
        else:
            component_info = COMPONENT_TASK_MAPPING.get(hypothesis.component)
        if pipeline:
            task_spec = T(f"scenarios.data_science.share:component_spec.Pipeline").r()
        elif DS_RD_SETTING.spec_enabled and sota_exp is not None:
            task_spec = sota_exp.experiment_workspace.file_dict[component_info["spec_file"]]
        else:
            task_spec = T(f"scenarios.data_science.share:component_spec.{hypothesis.component}").r()
        sys_prompt = T(".prompts_v2:task_gen.system").r(
            targets=component_info["target_name"],
            task_specification=task_spec,
            task_output_format=component_info["task_output_format"],
            component_desc=component_desc,
            workflow_check=not pipeline and hypothesis.component != "Workflow",
        )
        user_prompt = T(".prompts_v2:task_gen.user").r(
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            hypothesis=str(hypothesis),
            failed_exp_and_feedback_list_desc=failed_exp_feedback_list_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | Dict[str, str]],
        )
        task_dict = json.loads(response)
        task_design = task_dict.get("task_design", {})
        task_name = (
            task_design["model_name"] if (hypothesis.component == "Model" and not pipeline) else hypothesis.component
        )
        description = (
            task_design
            if isinstance(task_design, str)
            else task_design.get("description", f"{component_info['target_name']} description not provided")
        )
        task_class = component_info["task_class"]
        task = task_class(
            name=task_name,
            description=description,
        )
        new_workflow_desc = task_dict.get("workflow_update", "No update needed")
        exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)
        # exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
        if sota_exp is not None:
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)
        if not pipeline and new_workflow_desc != "No update needed":
            workflow_task = WorkflowTask(
                name="Workflow",
                description=new_workflow_desc,
            )
            exp.pending_tasks_list.append([workflow_task])
        return exp

    def gen(self, trace: DSTrace, pipeline: bool = False) -> DSExperiment:
        if pipeline:
            component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        else:
            component_desc = "\n".join(
                [
                    f"[{key}] {value}"
                    for key, value in T("scenarios.data_science.share:component_description").template.items()
                ]
            )

        sota_exp = trace.sota_experiment()
        if not isinstance(sota_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = sota_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)

        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="Best of previous exploration of the scenario"
        )

        exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="all"),
            type="all",
            pipeline=pipeline,
        )
        failed_exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="failed"),
            type="failed",
            pipeline=pipeline,
        )

        # Step 1: Identify problems
        all_problems = {}
        if len(trace.hist) >= 3:
            fb_problems = self.identify_feedback_problem(
                scenario_desc=scenario_desc,
                exp_feedback_list_desc=exp_feedback_list_desc,
                sota_exp_desc=sota_exp_desc,
            )
            for problem_name in fb_problems:
                fb_problems[problem_name]["label"] = "FEEDBACK_PROBLEM"
                all_problems[problem_name] = fb_problems[problem_name]

        if len(trace.hist) < 9:
            scen_problems = self.identify_scenario_problem(
                scenario_desc=scenario_desc,
                sota_exp_desc=sota_exp_desc,
            )
            for problem_name in scen_problems:
                scen_problems[problem_name]["label"] = "SCENARIO_PROBLEM"
                all_problems[problem_name] = scen_problems[problem_name]

        # Step 1.5: Sample ideas from idea pool
        if DS_RD_SETTING.enable_knowledge_base:
            all_problems = trace.knowledge_base.sample_ideas(
                problems=all_problems,
                scenario_desc=scenario_desc,
                exp_feedback_list_desc=exp_feedback_list_desc,
                sota_exp_desc=sota_exp_desc,
                competition_desc=self.scen.get_competition_full_desc(),
            )

        # Step 2: Propose hypothesis based on the identified problems (and sampled ideas)
        hypothesis_dict = self.hypothesis_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            exp_feedback_list_desc=exp_feedback_list_desc,
            sota_exp_desc=sota_exp_desc,
            problems=all_problems,
            pipeline=pipeline,
            enable_idea_pool=DS_RD_SETTING.enable_knowledge_base,
        )
        if not pipeline:
            sota_exp_model_file_count = len(
                [
                    k
                    for k in sota_exp.experiment_workspace.file_dict.keys()
                    if k.endswith(".py") and "test" not in k and k.startswith("model")
                ]
            )
            if sota_exp_model_file_count <= 1:
                pop_names = []
                for problem_name in hypothesis_dict:
                    if hypothesis_dict[problem_name].get("component", "") == "Ensemble":
                        pop_names.append(problem_name)
                for name in pop_names:
                    hypothesis_dict.pop(name)

        # Step 3: Select the best hypothesis
        pickled_problem_name, new_hypothesis = self.hypothesis_rank(
            hypothesis_dict=hypothesis_dict,
            problem_dict=all_problems,
            trace=trace,
        )
        # Step 3.5: Update knowledge base with the picked problem
        if DS_RD_SETTING.enable_knowledge_base:
            trace.knowledge_base.update_pickled_problem(all_problems, pickled_problem_name)

        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp,
            hypothesis=new_hypothesis,
            pipeline=pipeline,
            failed_exp_feedback_list_desc=failed_exp_feedback_list_desc,
        )
