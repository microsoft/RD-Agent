import json
from typing import Literal

import pandas as pd

from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.core.proposal import ExperimentFeedback, ExpGen, Hypothesis, Trace
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import COMPONENT, DSExperiment
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.utils.agent.tpl import T
from rdagent.utils.repo.diff import generate_diff_from_dict
from rdagent.utils.workflow import wait_retry


class DSHypothesis(Hypothesis):
    def __init__(
        self,
        component: COMPONENT,
        hypothesis: str = "",
        reason: str = "",
        concise_reason: str = "",
        concise_observation: str = "",
        concise_justification: str = "",
        concise_knowledge: str = "",
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.component = component

    def __str__(self) -> str:
        if self.hypothesis == "":
            return f"No hypothesis available. Trying to construct the first runnable {self.component} component."
        return f"""Chosen Component: {self.component}
Hypothesis: {self.hypothesis}
Reason: {self.reason}
Concise Reason & Knowledge: {self.concise_reason}
Concise Observation: {self.concise_observation}
Concise Justification: {self.concise_justification}
Concise Knowledge: {self.concise_knowledge}
"""


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
        "target_name": "Building model",
        "spec_file": "spec/model.md",
        "task_output_format": T(".prompts:output_format.model").r(),
        "task_class": ModelTask,
        "extra_params": {
            "model_type": "Model type not provided",
            "architecture": "Model architecture not provided",
            "hyperparameters": "Model hyperparameters not provided",
        },
        "extra_requirement": T(".prompts:extra_requirement.model").r(),
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
}


class DSTrace(Trace[DataScienceScen, KnowledgeBase]):
    def __init__(self, scen: DataScienceScen, knowledge_base: KnowledgeBase | None = None) -> None:
        self.scen: DataScienceScen = scen
        self.hist: list[tuple[DSExperiment, ExperimentFeedback]] = []
        self.knowledge_base = knowledge_base

    COMPLETE_ORDER = ("DataLoadSpec", "FeatureEng", "Model", "Ensemble", "Workflow")

    def next_incomplete_component(self) -> COMPONENT | None:
        """
        NOTE:
        - A component will be complete until get True decision feedback !!!
        """
        for c in self.COMPLETE_ORDER:
            if not self.has_component(c):
                return c
        return None

    def has_component(self, component: COMPONENT) -> bool:
        for exp, fb in self.hist:
            assert isinstance(exp.hypothesis, DSHypothesis), "Hypothesis should be DSHypothesis (and not None)"
            if exp.hypothesis.component == component and fb:
                return True
        return False

    def experiment_and_feedback_list_after_init(
        self, return_type: Literal["sota", "failed", "all"]
    ) -> list[tuple[DSExperiment, ExperimentFeedback]]:
        """
        Retrieve a list of experiments and feedbacks based on the return_type.

        Parameters
        ----------
        return_type : str
            One of "sota", "failed", "all".

        Returns
        -------
        list[tuple[DSExperiment, ExperimentFeedback]]
            List of experiments and feedbacks.
        """

        final_component = self.COMPLETE_ORDER[-1]
        has_final_component = False
        exp_and_feedback_list = []
        for exp, fb in self.hist:
            if has_final_component:
                if return_type == "all":
                    exp_and_feedback_list.append((exp, fb))
                elif return_type == "failed" and not fb.decision:
                    exp_and_feedback_list.append((exp, fb))
                elif return_type == "sota" and fb.decision:
                    exp_and_feedback_list.append((exp, fb))
            if exp.hypothesis.component == final_component and fb:
                has_final_component = True
        return exp_and_feedback_list

    def sota_experiment(self) -> DSExperiment | None:
        """
        Returns
        -------
        Experiment or None
            The experiment result if found, otherwise None.
        """
        if self.next_incomplete_component() is None:
            for exp, ef in self.hist[::-1]:
                # the sota exp should be accepted decision and all required components are completed.
                if ef.decision:
                    return exp
        return None

    def last_successful_exp(self) -> DSExperiment | None:
        """
        Access the last successful experiment even part of the components are not completed.
        """
        for exp, ef in self.hist[::-1]:
            if ef.decision:
                return exp
        return None

    def last_runnable_exp_fb(self) -> tuple[DSExperiment, ExperimentFeedback] | None:
        """
        Access the last runnable experiment (no exception, usually not all task failed) and feedback
        """
        for exp, ef in self.hist[::-1]:
            if ef.exception is None:
                return exp, ef
        return None


class DSExpGen(ExpGen):
    """Data Science Task Generator."""

    def __init__(self, scen: DataScienceScen, max_trace_hist: int = 3) -> None:
        self.max_trace_hist = max_trace_hist  # max number of historical trace to know when propose new experiment
        super().__init__(scen)

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
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
            )
        )

        return resp_dict

    def _handle_missing_component(
        self,
        component: COMPONENT,
        task_cls: type,
        scenario_desc: str,
        trace: Trace,
        last_successful_exp: DSExperiment | None,
        spec_file: str | None = None,
        component_prompt_key: str | None = None,
    ) -> DSExperiment:
        """Handle any component using a unified approach.

        Args:
            component: Name of the component (e.g. "DataLoadSpec")
            task_cls: The task class to instantiate (e.g. DataLoaderTask)
            scenario_desc: Description of the current scenario
            last_successful_exp: Last successful experiment or None
            spec_file: Path to specification file if needed
        """

        former_tasks_desc = ""
        if len(trace.hist) > 0:
            for exp, fb in reversed(trace.hist):
                if exp is not last_successful_exp:
                    former_task_desc = exp.pending_tasks_list[0][0].get_task_information()
                    former_task_desc += f"\n\nYou have tried to implement the same component and got the following exception: \n{fb.exception}\n Please try different methods to avoid the same errors and results in an infinite loop"
                    former_tasks_desc += former_task_desc
                else:
                    break

        resp_dict = self._init_task_gen(
            targets=component,
            scenario_desc=scenario_desc,
            spec=last_successful_exp.experiment_workspace.file_dict[spec_file] if spec_file else None,
            task_output_format=T(f".prompts:output_format.{component_prompt_key or component.lower()}").r(),
            former_task=former_tasks_desc if former_tasks_desc else None,
        )

        task = task_cls(
            name=component if component != "Model" else resp_dict.pop("model_name"),
            description=resp_dict.get("description", f"{component} description not provided"),
            **{
                k: resp_dict.get("extra_params", {}).get(k, v)
                for k, v in COMPONENT_TASK_MAPPING[component].get("extra_params", {}).items()
            },
        )

        exp = DSExperiment(pending_tasks_list=[[task]], hypothesis=DSHypothesis(component))
        if last_successful_exp:
            exp.experiment_workspace.inject_code_from_folder(last_successful_exp.experiment_workspace.workspace_path)
        return exp

    def gen(self, trace: DSTrace) -> DSExperiment:
        scenario_desc = trace.scen.get_scenario_all_desc()
        last_successful_exp = trace.last_successful_exp()

        next_missing_component = trace.next_incomplete_component()

        init_component_config = {
            "DataLoadSpec": {"task_cls": DataLoaderTask, "spec_file": None, "component_prompt_key": "data_loader"},
            "FeatureEng": {"task_cls": FeatureTask, "spec_file": "spec/feature.md", "component_prompt_key": "feature"},
            "Model": {"task_cls": ModelTask, "spec_file": "spec/model.md", "component_prompt_key": "model"},
            "Ensemble": {"task_cls": EnsembleTask, "spec_file": "spec/ensemble.md", "component_prompt_key": "ensemble"},
            "Workflow": {"task_cls": WorkflowTask, "spec_file": "spec/workflow.md", "component_prompt_key": "workflow"},
        }

        if next_missing_component in init_component_config:
            # TODO: we may merge the if else logic in the future.
            # the current
            config = init_component_config[next_missing_component]
            return self._handle_missing_component(
                component=next_missing_component,
                task_cls=config["task_cls"],
                scenario_desc=scenario_desc,
                last_successful_exp=last_successful_exp,
                spec_file=config.get("spec_file"),
                trace=trace,
                component_prompt_key=config.get("component_prompt_key"),
            )
        else:  # propose new component by LLM
            # Guidelines:
            # System prompts: Shared condition you are facing
            # - scenario description: `scenario_desc`
            # - expected output format
            # User prompts: Task Specific information
            # - Previous Feedback
            # - Current sota implementation (encourage change based on it)
            # - Extra RAG
            sota_exp = trace.sota_experiment()
            assert sota_exp is not None, "SOTA experiment is not provided."
            exp_and_feedback = trace.hist[-1]
            last_exp = exp_and_feedback[0]

            # Step 1: Generate component
            # Describe current best solution using shared template
            sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
                exp=sota_exp, heading="Best of previous exploration of the scenario"
            )
            last_exp_diff = "\n".join(
                generate_diff_from_dict(
                    sota_exp.experiment_workspace.file_dict, last_exp.experiment_workspace.file_dict
                )
            )  # we use file_dict for hitting the cache when replicate the experiment in another machine.

            sota_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="sota")
            failed_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="failed")[
                -self.max_trace_hist :
            ]
            all_exp_feedback_list = trace.experiment_and_feedback_list_after_init(return_type="all")
            trace_component_to_feedback_df = pd.DataFrame(columns=["component", "hypothesis", "decision"])
            for index, (exp, fb) in enumerate(all_exp_feedback_list):
                trace_component_to_feedback_df.loc[f"trial {index + 1}"] = [
                    exp.hypothesis.component,
                    exp.hypothesis.hypothesis,
                    fb.decision,
                ]

            sota_exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
                exp_and_feedback_list=sota_exp_feedback_list,
                success=True,
            )
            failed_exp_feedback_list_desc = T("scenarios.data_science.share:describe.trace").r(
                exp_and_feedback_list=failed_exp_feedback_list,
                success=False,
            )

            # Generate component using template with proper context
            component_sys_prompt = T(".prompts:component_gen.system").r(
                scenario=scenario_desc,
                sota_exp_desc=sota_exp_desc,
                last_exp_diff=last_exp_diff,
                component_output_format=T(".prompts:output_format.component").r(),
            )

            component_user_prompt = T(".prompts:component_gen.user").r(
                sota_exp_and_feedback_list_desc=sota_exp_feedback_list_desc,
                failed_exp_and_feedback_list_desc=failed_exp_feedback_list_desc,
                component_and_feedback_df=(
                    trace_component_to_feedback_df.to_string()
                    if len(trace_component_to_feedback_df) > 0
                    else "No experiment and feedback provided"
                ),
            )

            resp_dict_component: dict = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    component_user_prompt, component_sys_prompt, json_mode=True
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
                system_prompt = T(".prompts:direct_exp_gen.system").r(
                    targets=component_info["target_name"],
                    component=component,
                    scenario=scenario_desc,
                    hypothesis_specification=T(".prompts:hypothesis_specification").r(),
                    hypothesis_output_format=T(".prompts:output_format.hypothesis").r(),
                    task_specification=sota_exp.experiment_workspace.file_dict[component_info["spec_file"]],
                    task_output_format=component_info["task_output_format"],
                    extra_requirement=component_info.get("extra_requirement"),
                    workflow_check=(not component == "Workflow"),
                )

                user_prompt = T(".prompts:direct_exp_gen.user").r(
                    targets=component_info["target_name"],
                    sota_exp_and_feedback_list_desc=sota_exp_feedback_list_desc,
                    failed_exp_and_feedback_list_desc=failed_exp_feedback_list_desc,
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
                            user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
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
                exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)

                if new_workflow_desc != "No update needed":
                    workflow_task = WorkflowTask(
                        name="Workflow",
                        description=new_workflow_desc,
                    )
                    exp.pending_tasks_list.append([workflow_task])
                return exp
            else:
                raise ValueError(f"Unknown component: {component}")
