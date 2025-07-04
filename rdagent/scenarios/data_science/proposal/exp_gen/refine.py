import json
from typing import Any, Dict, List, Optional, Tuple

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExpGen
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.utils import (
    CodingSketch,
    get_component,
)
from rdagent.utils.agent.tpl import T


class DSRefineExpGen(ExpGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_function_calling = APIBackend().support_function_calling()

    def task_gen(
        self,
        component_desc: str,
        scenario_desc: str,
        sota_exp_desc: str,
        sota_exp: DSExperiment,
        hypothesis: DSHypothesis,
        exp_and_feedback_list_desc: str,
    ) -> DSExperiment:
        component_info = get_component("Pipeline")
        data_folder_info = self.scen.processed_data_folder_description
        sys_prompt = T(".prompts_refine:task_gen.system").r(
            task_output_format=component_info["task_output_format"] if not self.support_function_calling else None,
            component_desc=component_desc,
        )
        user_prompt = T(".prompts_refine:task_gen.user").r(
            scenario_desc=scenario_desc,
            data_folder_info=data_folder_info,
            sota_exp_desc=sota_exp_desc,
            hypothesis=hypothesis,
            exp_and_feedback_list_desc=exp_and_feedback_list_desc,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            response_format=CodingSketch if self.support_function_calling else {"type": "json_object"},
            json_target_type=Dict[str, str | Dict[str, str]] if not self.support_function_calling else None,
        )
        task_dict = json.loads(response)
        task_design = (
            task_dict.get("task_design", {}) if not self.support_function_calling else task_dict.get("sketch", {})
        )
        logger.info(f"Task design:\n{task_design}")
        task_name = hypothesis.component
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
        exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)
        if sota_exp is not None:
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)
        return exp

    def gen(self, trace: DSTrace) -> DSExperiment:
        # Step 0: Prepare
        pipeline = DS_RD_SETTING.coder_on_whole_pipeline
        component_desc = T("scenarios.data_science.share:component_description_in_pipeline").r()
        sota_exp, sota_fb = trace.sota_experiment_fb()
        if not isinstance(sota_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = sota_exp.experiment_workspace.file_dict.get("EDA.md", None)
        scenario_desc = self.scen.get_scenario_all_desc(trace=trace, eda_output=eda_output)
        sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp, heading="Best of previous exploration of the scenario"
        )
        exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="all"),
            type="all",
            pipeline=pipeline,
        )

        # Step 1: Generate a Pseudo Hypothesis for Refinement
        hypothesis = DSHypothesis(
            component="Model",
            hypothesis="The current pipeline is note effective in terms of efficiency or hyperparameters. Refinement or Adjustment should be made.",
            reason=sota_fb.reason,
        )

        # Step 2: Design Task to Refine SOTA
        return self.task_gen(
            component_desc=component_desc,
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp,
            hypothesis=hypothesis,
            exp_and_feedback_list_desc=exp_feedback_list_desc,
        )
