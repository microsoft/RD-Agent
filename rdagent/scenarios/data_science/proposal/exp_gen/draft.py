import json
from typing import TYPE_CHECKING, Dict

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.proposal import ExpGen, Hypothesis
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import COMPONENT, DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.utils.agent.tpl import T


class DSDraftExpGen(ExpGen):

    def _init_task_gen(
        self,
        targets: str,
        scenario_desc: str,
        task_output_format: str,
        workspace_code: str | None = None,
        spec: str = None,
        hypothesis: Hypothesis | None = None,
        exp_and_feedback_desc: str | None = None,
        former_task: str | None = None,
    ) -> dict:
        system_prompt = T(".prompts:task_gen.system").r(
            targets=targets,
            scenario=scenario_desc,
            task_specification=spec,
            hypothesis=hypothesis,
            task_output_format=task_output_format,
        )
        user_prompt = T(".prompts:task_gen.user").r(
            targets=targets,
            hypothesis=hypothesis,
            workspace_code=workspace_code,
            exp_and_feedback_desc=exp_and_feedback_desc,
            former_task_desc=former_task,
        )

        resp_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True, json_target_type=dict
            )
        )

        return resp_dict

    def gen(
        self,
        component: COMPONENT,
        trace: DSTrace,
    ) -> DSExperiment:
        """Handle any component using a unified approach.

        Args:
            component: Name of the component (e.g. "DataLoadSpec")
            task_cls: The task class to instantiate (e.g. DataLoaderTask)
            scenario_desc: Description of the current scenario
            last_successful_exp: Last successful experiment or None
            spec_file: Path to specification file if needed
            selection: The selection of the node to generate the task
        """
        last_successful_exp = trace.last_successful_exp()
        # typecheck on the last successful exp, should be DSExperiment
        if not isinstance(last_successful_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = last_successful_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)
        init_component_config = {
            "DataLoadSpec": {"task_cls": DataLoaderTask, "spec_file": None, "component_prompt_key": "data_loader"},
            "FeatureEng": {"task_cls": FeatureTask, "spec_file": "spec/feature.md", "component_prompt_key": "feature"},
            "Model": {"task_cls": ModelTask, "spec_file": "spec/model.md", "component_prompt_key": "model"},
            "Ensemble": {"task_cls": EnsembleTask, "spec_file": "spec/ensemble.md", "component_prompt_key": "ensemble"},
            "Workflow": {"task_cls": WorkflowTask, "spec_file": "spec/workflow.md", "component_prompt_key": "workflow"},
        }
        task_cls = init_component_config[component]["task_cls"]
        spec_file = init_component_config[component].get("spec_file")
        component_prompt_key = init_component_config[component].get("component_prompt_key")

        former_tasks_desc = ""
        search_list = trace.retrieve_search_list()
        if len(search_list) > 0:
            for exp, fb in reversed(search_list):
                if exp is not last_successful_exp:
                    former_task_desc = exp.pending_tasks_list[0][0].get_task_information()
                    former_task_desc += f"\n\nYou have tried to implement the same component and got the following exception: \n{fb.exception}\n Please try different methods to avoid the same errors and results in an infinite loop"
                    former_tasks_desc += former_task_desc
                else:
                    break

        if DS_RD_SETTING.spec_enabled:
            spec = last_successful_exp.experiment_workspace.file_dict[spec_file] if spec_file else None
        else:
            spec = T(f"scenarios.data_science.share:component_spec.{component}").r()
        resp_dict = self._init_task_gen(
            targets=component,
            scenario_desc=scenario_desc,
            spec=spec,
            task_output_format=T(f".prompts:output_format.{component_prompt_key or component.lower()}").r(),
            former_task=former_tasks_desc if former_tasks_desc else None,
        )

        task = task_cls(
            name=component if component != "Model" else resp_dict.pop("model_name"),
            description=resp_dict.get("description", f"{component} description not provided"),
        )

        exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=DSHypothesis(component))
        if last_successful_exp:
            # exp.experiment_workspace.inject_code_from_folder(last_successful_exp.experiment_workspace.workspace_path)
            exp.experiment_workspace.inject_code_from_file_dict(last_successful_exp.experiment_workspace)
        return exp


class DSDraftV2ExpGen(ExpGen):
    def task_gen(
        self,
        scenario_desc: str,
        scen_problems: dict,
        drafting_trace_desc: str,
    ) -> DSExperiment:
        scen_problems_text = ""
        for i, (problem_name, problem_dict) in enumerate(scen_problems.items()):
            scen_problems_text += f"## Problem Name: {problem_name}\n"
            scen_problems_text += f"- Problem Description: {problem_dict['problem']}\n\n"
        sys_prompt = T(".prompts_drafting:task_draft.system").r(
            task_spec=T(f"scenarios.data_science.share:component_spec.Pipeline").r(),
        )
        user_prompt = T(".prompts_drafting:task_draft.user").r(
            scenario_desc=scenario_desc,
            scen_problems=scen_problems_text,
            drafting_trace_desc=drafting_trace_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str],
        )
        task_dict = json.loads(response)
        task_design = task_dict.get("task_design", "Description not provided")
        task = PipelineTask(name="Workflow", description=task_design)

        # we use a pesudo hypothesis here
        pseudo_hypothesis = DSHypothesis(
            component="Workflow",
            hypothesis="This is a pseudo hypothesis for drafting the first competition implementation. Your result should not be influenced by this hypothesis.",
            problem_name="This is a pseudo problem name for drafting. The corresponding problem description includes several problem together.",
            problem_desc=scen_problems_text,
        )
        exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=pseudo_hypothesis)
        return exp

    def gen(self, trace: DSTrace) -> DSExperiment:
        # Prepare
        last_exp = trace.last_exp()
        if not isinstance(last_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = last_exp.experiment_workspace.file_dict.get("EDA.md", None)

        component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)
        drafting_trace_desc = T("scenarios.data_science.share:describe.drafting_trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="all"),
        )

        # Step 1: Identify Scenario Problems
        sys_prompt = T(".prompts_drafting:scenario_problem.system").r()
        user_prompt = T(".prompts_drafting:scenario_problem.user").r(scenario_desc=scenario_desc)
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, Dict[str, str]],
        )
        scen_problems = json.loads(response)

        # Step 2: Design Task
        return self.task_gen(
            scenario_desc=scenario_desc,
            scen_problems=scen_problems,
            drafting_trace_desc=drafting_trace_desc,
        )
