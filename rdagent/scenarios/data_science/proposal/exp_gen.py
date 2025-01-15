import json
import re
from typing import Literal

import pandas as pd

from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.experiment import Experiment, Workspace
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.core.proposal import (
    ExperimentFeedback,
    ExpGen,
    Hypothesis,
    HypothesisFeedback,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import COMPONENT, DSExperiment
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.utils.agent.tpl import T


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
            return f"Chosen Component: {self.component}"
        return f"""Chosen Component: {self.component}
Hypothesis: {self.hypothesis}
Reason: {self.reason}
Concise Reason & Knowledge: {self.concise_reason}
Concise Observation: {self.concise_observation}
Concise Justification: {self.concise_justification}
Concise Knowledge: {self.concise_knowledge}
"""


class DSTrace(Trace[DataScienceScen, KnowledgeBase]):
    def __init__(self, scen: DataScienceScen, knowledge_base: KnowledgeBase | None = None) -> None:
        self.scen: DataScienceScen = scen
        self.hist: list[tuple[DSExperiment, ExperimentFeedback]] = []
        self.knowledge_base = knowledge_base

    def sota_experiment(self, last_n: int = -1) -> DSExperiment | None:
        """
        Access the last experiment result.

        Parameters
        ----------
        last_n : int
            The index from the last experiment result to access.
            Use -1 for the most recent experiment, -2 for the second most recent, and so on.

        Returns
        -------
        Experiment or None
            The experiment result if found, otherwise None.
        """
        assert last_n < 0
        for exp, ef in self.hist[::-1]:
            # the sota exp should be accepted decision and all required components are completed.
            if ef.decision and exp.next_component_required() is None:
                last_n += 1
                if last_n == 0:
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


class DSExpGen(ExpGen):
    """Data Science Task Generator."""

    def llm_task_gen(
        self,
        targets: str,
        scenario_desc: str,
        task_output_format: str,
        workspace_code: str | None = None,
        spec: str = None,
        hypothesis: Hypothesis | None = None,
        exp_and_feedback_desc: str | None = None,
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
        resp_dict = self.llm_task_gen(
            targets=component,
            scenario_desc=scenario_desc,
            spec=last_successful_exp.experiment_workspace.file_dict[spec_file] if spec_file else None,
            task_output_format=T(f".prompts:output_format.{component_prompt_key or component.lower()}").r(),
        )

        # Create task instance
        exp_and_feedback = trace.hist[-1] if len(trace.hist) > 0 else None
        if (
            exp_and_feedback
            and exp_and_feedback[1].exception is not None
            and (
                exp_and_feedback[0].sub_tasks[0].name == component
                or exp_and_feedback[0].sub_tasks[0].name.startswith("model_")
                and component == "Model"
            )
        ):  # Assumption: when completing missing component, using component name as task name
            resp_dict[
                "description"
            ] += f"\n\nYou have tried to implement the same component and got the following exception: \n{exp_and_feedback[1].exception}\n Please try different methods to avoid the same errors and results in an infinite loop"

        task = task_cls(
            name=component if component != "Model" else resp_dict.pop("model_name"),
            **resp_dict,
        )

        exp = DSExperiment(sub_tasks=[task], hypothesis=DSHypothesis(component))
        if last_successful_exp:
            exp.experiment_workspace.inject_code_from_folder(last_successful_exp.experiment_workspace.workspace_path)
        return exp

    def gen(self, trace: DSTrace) -> DSExperiment:
        scenario_desc = trace.scen.get_scenario_all_desc()
        last_successful_exp = trace.last_successful_exp()

        if len(trace.hist) == 0 or last_successful_exp is None:
            next_missing_component = "DataLoadSpec"
        else:
            next_missing_component = last_successful_exp.next_component_required()

        component_config = {
            "DataLoadSpec": {"task_cls": DataLoaderTask, "spec_file": None, "component_prompt_key": "data_loader"},
            "FeatureEng": {"task_cls": FeatureTask, "spec_file": "spec/feature.md", "component_prompt_key": "feature"},
            "Model": {"task_cls": ModelTask, "spec_file": "spec/model.md", "component_prompt_key": "model"},
            "Ensemble": {"task_cls": EnsembleTask, "spec_file": "spec/ensemble.md", "component_prompt_key": "ensemble"},
            "Workflow": {"task_cls": WorkflowTask, "spec_file": "spec/workflow.md", "component_prompt_key": "workflow"},
        }

        if next_missing_component in component_config:
            config = component_config[next_missing_component]
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
            assert last_successful_exp is not None, "SOTA experiment is not provided."
            exp_and_feedback = trace.hist[-1]
            last_exp = exp_and_feedback[0]

            # Step 1: Generate component
            # Describe current best solution using shared template
            sota_solution = trace.sota_experiment()
            sota_exp_desc = T("scenarios.data_science.share:describe.exp").r(
                exp=last_successful_exp, heading="Best of previous exploration of the scenario"
            )
            current_exp_desc = T("scenarios.data_science.share:describe.exp").r(
                exp=last_exp, heading="Current exploration of the scenario"
            )
            exp_and_feedback_desc = T("scenarios.data_science.share:describe.feedback").r(
                exp_and_feedback=exp_and_feedback
            )

            # Generate component using template with proper context
            component_sys_prompt = T(".prompts:component_gen.system").r(
                scenario=scenario_desc,
                sota_exp_desc=sota_exp_desc,
                current_exp_desc=current_exp_desc,
                component_output_format=T(".prompts:output_format.component").r(),
            )

            component_user_prompt = T(".prompts:component_gen.user").r(
                exp_and_feedback_desc=exp_and_feedback_desc,
            )

            resp_dict_component: dict = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    component_user_prompt, component_sys_prompt, json_mode=True
                )
            )

            component = resp_dict_component.get("component", "Component not provided")

            # Why we should split component selection and hypothesis generation
            # - after we know the selected component, we can use RAG.

            # Step 2: Generate the rest of the hypothesis
            if component != "Model":
                hypothesis_sys_prompt = T(".prompts:hypothesis_gen.system").r(
                    targets="data science project",
                    scenario=scenario_desc,
                    hypothesis_output_format=T(".prompts:output_format.hypothesis").r(),
                    hypothesis_specification=T(".prompts:hypothesis_specification").r(sota_solution=sota_solution),
                    component=component,
                )
                hypothesis_user_prompt = T(".prompts:hypothesis_gen.user").r(
                    targets="data science project",
                    exp_and_feedback_desc=exp_and_feedback_desc,
                )

                resp_dict: dict = json.loads(
                    APIBackend().build_messages_and_create_chat_completion(
                        hypothesis_user_prompt, hypothesis_sys_prompt, json_mode=True
                    )
                )
                hypothesis = DSHypothesis(
                    component=resp_dict.get("component", "Component not provided"),
                    hypothesis=resp_dict.get("hypothesis", "Hypothesis not provided"),
                    reason=resp_dict.get("reason", "Reason not provided"),
                    concise_reason=resp_dict.get("concise_reason", "Concise reason not provided"),
                    concise_observation=resp_dict.get("concise_observation", "Concise observation not provided"),
                    concise_justification=resp_dict.get("concise_justification", "Concise justification not provided"),
                    concise_knowledge=resp_dict.get("concise_knowledge", "Concise knowledge not provided"),
                )
            else:
                model_infos = []
                score_df = pd.read_csv(
                    last_successful_exp.experiment_workspace.workspace_path / "scores.csv", index_col=0
                )
                metric_name = score_df.columns[0]
                for fname in last_successful_exp.experiment_workspace.file_dict:
                    if re.match(r"^model_(?!test)\w+\.py$", fname):
                        model_str = f"{fname}:\n{metric_name} on valid: {score_df.loc[fname[:-3]]}\n```python\n{last_successful_exp.experiment_workspace.file_dict[fname]}\n```\n"
                        model_infos.append(model_str)

                model_num = len(model_infos)
                models_info_str = ("-" * 20).join(model_infos)
                if model_num >= 3:
                    hypothesis_sys_prompt = T(".prompts:hypothesis_model.system").r(
                        targets="data science project",
                        scenario=scenario_desc,
                        hypothesis_output_format=T(".prompts:output_format.hypothesis").r(),
                        hypothesis_specification=T(".prompts:hypothesis_specification").r(sota_solution=sota_solution),
                        model_info=models_info_str,
                        model_enough=True,
                    )
                else:
                    hypothesis_sys_prompt = T(".prompts:hypothesis_model.system").r(
                        targets="data science project",
                        scenario=scenario_desc,
                        hypothesis_output_format=T(".prompts:output_format.hypothesis").r(),
                        hypothesis_specification=T(".prompts:hypothesis_specification").r(sota_solution=sota_solution),
                        model_info=models_info_str,
                        model_enough=False,
                    )
                hypothesis_user_prompt = T(".prompts:hypothesis_gen.user").r(
                    targets="data science project",
                    exp_and_feedback_desc=exp_and_feedback_desc,
                )
                resp_dict: dict = json.loads(
                    APIBackend().build_messages_and_create_chat_completion(
                        hypothesis_user_prompt, hypothesis_sys_prompt, json_mode=True
                    )
                )
                hypothesis = DSHypothesis(
                    component=resp_dict.get("component", "Component not provided"),
                    hypothesis=resp_dict.get("hypothesis", "Hypothesis not provided"),
                    reason=resp_dict.get("reason", "Reason not provided"),
                    concise_reason=resp_dict.get("concise_reason", "Concise reason not provided"),
                    concise_observation=resp_dict.get("concise_observation", "Concise observation not provided"),
                    concise_justification=resp_dict.get("concise_justification", "Concise justification not provided"),
                    concise_knowledge=resp_dict.get("concise_knowledge", "Concise knowledge not provided"),
                )

            # 2. gen experiment
            if hypothesis.component == "DataLoadSpec":
                resp_dict = self.llm_task_gen(
                    targets="Data loader and specification generation",
                    scenario_desc=scenario_desc,
                    spec=last_successful_exp.experiment_workspace.file_dict["spec/data_loader.md"],
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.data_loader").r(),
                    exp_and_feedback_desc=exp_and_feedback_desc,
                )

                dt = DataLoaderTask(
                    name="Data loader and specification generation",
                    description=resp_dict.get(
                        "description", "Data loader and specification generation description not provided"
                    ),
                )

                exp = DSExperiment(sub_tasks=[dt], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(
                    last_successful_exp.experiment_workspace.workspace_path
                )
                return exp
            elif hypothesis.component == "FeatureEng":
                # TODO: RAG
                resp_dict = self.llm_task_gen(
                    targets="Feature Engineering",
                    scenario_desc=scenario_desc,
                    spec=last_successful_exp.experiment_workspace.file_dict["spec/feature.md"],
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.feature").r(),
                    exp_and_feedback_desc=exp_and_feedback_desc,
                )

                ft = FeatureTask(
                    name="Feature Engineering",
                    description=resp_dict.get("description", "Feature description not provided"),
                )

                exp = DSExperiment(sub_tasks=[ft], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(
                    last_successful_exp.experiment_workspace.workspace_path
                )
                return exp
            elif hypothesis.component == "Model":
                resp_dict = self.llm_task_gen(
                    targets="Models",
                    scenario_desc=scenario_desc,
                    spec=last_successful_exp.experiment_workspace.file_dict["spec/model.md"],
                    hypothesis=hypothesis,
                    workspace_code=last_successful_exp.experiment_workspace.all_codes,
                    task_output_format=T(".prompts:output_format.model").r(),
                    exp_and_feedback_desc=exp_and_feedback_desc,
                )

                mt = ModelTask(
                    name=resp_dict.get("model_name", "Model name not provided"),
                    description=resp_dict.get("description", "Model description not provided"),
                    model_type=resp_dict.get("model_type", "Model type not provided"),
                    architecture=resp_dict.get("architecture", "Model architecture not provided"),
                    hyperparameters=resp_dict.get("hyperparameters", "Model hyperparameters not provided"),
                    base_code="",
                )

                exp = DSExperiment(sub_tasks=[mt], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(
                    last_successful_exp.experiment_workspace.workspace_path
                )
                return exp
            elif hypothesis.component == "Ensemble":
                resp_dict = self.llm_task_gen(
                    targets="Ensemble",
                    scenario_desc=scenario_desc,
                    spec=last_successful_exp.experiment_workspace.file_dict["spec/ensemble.md"],
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.ensemble").r(),
                    exp_and_feedback_desc=exp_and_feedback_desc,
                )

                et = EnsembleTask(
                    name="Ensemble",
                    description=resp_dict.get("description", "Ensemble description not provided"),
                )

                exp = DSExperiment(sub_tasks=[et], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(
                    last_successful_exp.experiment_workspace.workspace_path
                )
                return exp
            elif hypothesis.component == "Workflow":
                resp_dict = self.llm_task_gen(
                    targets="Workflow",
                    scenario_desc=scenario_desc,
                    spec=last_successful_exp.experiment_workspace.file_dict["spec/workflow.md"],
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.workflow").r(),
                    exp_and_feedback_desc=exp_and_feedback_desc,
                )

                wt = WorkflowTask(
                    name="Workflow",
                    description=resp_dict.get("description", "Workflow description not provided"),
                )

                exp = DSExperiment(sub_tasks=[wt], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(
                    last_successful_exp.experiment_workspace.workspace_path
                )
                return exp

        return super().gen(trace)
