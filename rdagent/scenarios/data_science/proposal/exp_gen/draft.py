import json
from typing import TYPE_CHECKING
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.proposal import ExpGen, Hypothesis
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import COMPONENT, DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.utils.agent.tpl import T
from rdagent.log import rdagent_logger as logger
# from rdagent.scenarios.data_science.proposal.exp_gen.proposal import get_component
from typing import Any, Dict
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask

_COMPONENT_META: Dict[str, Dict[str, Any]] = {
    "DataLoadSpec": {
        "target_name": "Data loader and specification generation",
        "spec_file": "spec/data_loader.md",
        "output_format_key": ".prompts:output_format.data_loader",
        "task_class": DataLoaderTask,
    },
    "FeatureEng": {
        "target_name": "Feature engineering",
        "spec_file": "spec/feature.md",
        "output_format_key": ".prompts:output_format.feature",
        "task_class": FeatureTask,
    },
    "Model": {
        "target_name": "Model",
        "spec_file": "spec/model.md",
        "output_format_key": ".prompts:output_format.model",
        "task_class": ModelTask,
    },
    "Ensemble": {
        "target_name": "Ensemble",
        "spec_file": "spec/ensemble.md",
        "output_format_key": ".prompts:output_format.ensemble",
        "task_class": EnsembleTask,
    },
    "Workflow": {
        "target_name": "Workflow",
        "spec_file": "spec/workflow.md",
        "output_format_key": ".prompts:output_format.workflow",
        "task_class": WorkflowTask,
    },
    "Pipeline": {
        "target_name": "Pipeline",
        "spec_file": None,
        "output_format_key": ".prompts:output_format.pipeline",
        "task_class": PipelineTask,
    },
}


def get_component(name: str) -> Dict[str, Any]:
    meta = _COMPONENT_META.get(name)
    if meta is None:
        raise KeyError(f"Unknown component: {name!r}")

    return {
        "target_name": meta["target_name"],
        "spec_file": meta["spec_file"],
        "task_output_format": T(meta["output_format_key"]).r(),
        "task_class": meta["task_class"],
    }

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


class DSDraftExpGenV2(ExpGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_function_calling = APIBackend().support_function_calling()


    def tag_gen(self, scenario_desc: str) -> str:
        sys_prompt = T(".prompts_draft:tag_gen.system").r(
            tag_desc=T(".prompts_draft:description.tag_description").r()
        )
        user_prompt = T(".prompts_draft:tag_gen.user").r(
            scenario_desc=scenario_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str],
        )
        return json.loads(response)['tag'].lower()
    
    def knowledge_gen(self) -> str:
        general_knowledge = T(".prompts_draft:knowledge.general").r()
        return f"{general_knowledge}"

    def hypothesis_gen(
        self,
        knowledge: str,
        component_desc: str,
        scenario_desc: str,
        failed_exp_feedback_list_desc: str,
    ) -> DSHypothesis:
        sys_prompt = T(".prompts_draft:hypothesis_draft.system").r(
            component_desc=component_desc
        )
        user_prompt = T(".prompts_draft:hypothesis_draft.user").r(
            scenario_desc=scenario_desc,
            knowledge=knowledge,
            failed_exp_feedback_list_desc=failed_exp_feedback_list_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str],
        )
        resp_dict = json.loads(response)
        return DSHypothesis(
            component=resp_dict.get("component", "Model"),
            hypothesis=resp_dict.get("hypothesis", "Hypothesis not provided"),
            reason=resp_dict.get("reason", "Reason not provided"),
        )

    def task_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        hypotheses: list[DSHypothesis],
        pipeline: bool,
        knowledge: str,
        failed_exp_feedback_list_desc: str,
    ) -> DSExperiment:
        if pipeline:
            component_info = get_component("Pipeline")
        else:
            component_info = get_component(hypotheses[0].component)
        data_folder_info = self.scen.processed_data_folder_description
        sys_prompt = T(".prompts_draft:task_gen.system").r(
            task_output_format=component_info["task_output_format"],
            component_desc=component_desc,
            workflow_check=not pipeline and hypotheses[0].component != "Workflow",
        )
        user_prompt = T(".prompts_draft:task_gen.user").r(
            scenario_desc=scenario_desc,
            knowledge=knowledge,
            data_folder_info=data_folder_info,
            hypothesis=hypotheses[0], # FIXME: pass 1 hypothesis only
            failed_exp_and_feedback_list_desc=failed_exp_feedback_list_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | Dict[str, str]],
        )
        task_dict = json.loads(response)
        task_design = (
            task_dict.get("task_design", {}) if not self.support_function_calling else task_dict.get("sketch", {})
        )
        logger.info(f"Task design:\n{task_design}")
        task_name = hypotheses[0].component
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
        exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypotheses[0])
        if not pipeline and new_workflow_desc != "No update needed":
            workflow_task = WorkflowTask(
                name="Workflow",
                description=new_workflow_desc,
            )
            exp.pending_tasks_list.append([workflow_task])
        return exp

    def gen(self, trace: DSTrace) -> DSExperiment:
        # Step 0: Prepare
        pipeline = DS_RD_SETTING.coder_on_whole_pipeline
        if pipeline:
            component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        else:
            component_desc = "\n".join(
                [
                    f"[{key}] {value}"
                    for key, value in T("scenarios.data_science.share:component_description").template.items()
                ]
            )

        last_exp = trace.last_exp()
        if not isinstance(last_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = last_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=eda_output)
        
        failed_exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="failed"),
            type="failed",
            pipeline=pipeline,
        )

        # Step 1: Generate Tags TODO: do this part in the scenario analysis part
        # tag = self.tag_gen(scenario_desc)
        knowledge = self.knowledge_gen()

        # Step 2: Generate Hypothesis based on General Knowledge
        hypothesis = self.hypothesis_gen(
            knowledge=knowledge,
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            failed_exp_feedback_list_desc=failed_exp_feedback_list_desc,
        )

        # Step 3: Design Task
        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            hypotheses=[hypothesis],
            failed_exp_feedback_list_desc=failed_exp_feedback_list_desc,
            knowledge=knowledge,
            pipeline=pipeline,
        )