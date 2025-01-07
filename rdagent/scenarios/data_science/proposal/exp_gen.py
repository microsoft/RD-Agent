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

    def sota_experiment(self) -> Experiment | None:
        """Access the last experiment result."""
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
        hypothesis_and_feedback: str | None = None,
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
            hypothesis_and_feedback=hypothesis_and_feedback,
        )

        resp_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
            )
        )

        return resp_dict

    def gen(self, trace: DSTrace) -> DSExperiment:
        scenario_desc = trace.scen.get_scenario_all_desc()
        sota_exp = trace.sota_experiment()

        if len(trace.hist) == 0:
            next_component = "DataLoadSpec"
        else:
            next_component = sota_exp.next_component_required()

        if next_component == "DataLoadSpec":
            resp_dict = self.llm_task_gen(
                targets="Data loader and specification generation",
                scenario_desc=scenario_desc,
                task_output_format=T(".prompts:output_format.data_loader").r(),
            )
            dt = DataLoaderTask(
                name="Data loader and specification generation",
                description=resp_dict.get(
                    "description", "Data loader and specification generation description not provided"
                ),
            )

            exp = DSExperiment(sub_tasks=[dt], hypothesis=DSHypothesis("DataLoadSpec"))
            return exp
        elif next_component == "FeatureEng":
            resp_dict = self.llm_task_gen(
                targets="Feature Engineering",
                scenario_desc=scenario_desc,
                spec=sota_exp.experiment_workspace.file_dict["spec/feature.md"],
                task_output_format=T(".prompts:output_format.feature").r(),
            )

            ft = FeatureTask(
                name="Feature Engineering",
                description=resp_dict.get("description", "Factor description not provided"),
            )
            exp = DSExperiment(sub_tasks=[ft], hypothesis=DSHypothesis("FeatureEng"))
            exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
            return exp
        elif next_component == "Model":
            resp_dict = self.llm_task_gen(
                targets="Models",
                scenario_desc=scenario_desc,
                spec=sota_exp.experiment_workspace.file_dict["spec/model.md"],
                task_output_format=T(".prompts:output_format.model").r(),
            )

            mt = ModelTask(
                name=resp_dict.get("model_name", "Model name not provided"),
                description=resp_dict.get("description", "Model description not provided"),
                model_type=resp_dict.get("model_type", "Model type not provided"),
                architecture=resp_dict.get("architecture", "Model architecture not provided"),
                hyperparameters=resp_dict.get("hyperparameters", "Model hyperparameters not provided"),
            )
            exp = DSExperiment(sub_tasks=[mt], hypothesis=DSHypothesis("Model"))
            exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
            return exp
        elif next_component == "Ensemble":
            resp_dict = self.llm_task_gen(
                targets="Ensemble",
                scenario_desc=scenario_desc,
                spec=sota_exp.experiment_workspace.file_dict["spec/ensemble.md"],
                task_output_format=T(".prompts:output_format.ensemble").r(),
            )

            et = EnsembleTask(
                name="Ensemble",
                description=resp_dict.get("description", "Ensemble description not provided"),
            )
            exp = DSExperiment(sub_tasks=[et], hypothesis=DSHypothesis("Ensemble"))
            exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
            return exp
        elif next_component == "Workflow":
            resp_dict = self.llm_task_gen(
                targets="Workflow",
                scenario_desc=scenario_desc,
                spec=sota_exp.experiment_workspace.file_dict["spec/workflow.md"],
                task_output_format=T(".prompts:output_format.workflow").r(),
            )

            wt = WorkflowTask(
                name="Workflow",
                description=resp_dict.get("description", "Workflow description not provided"),
            )
            exp = DSExperiment(sub_tasks=[wt], hypothesis=DSHypothesis("Workflow"))
            exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
            return exp
        else:  # propose new component by LLM
            assert sota_exp is not None, "SOTA experiment is not provided."

            # base info
            hypothesis_and_feedback = T(".prompts:hypothesis_and_feedback").r(trace=trace)
            # Step 1: Generate component
            sota_solution = ""
            component_sys_prompt = T(".prompts:component_gen").r(
                targets="data science project",
                scenario=scenario_desc,
                hypothesis_output_format=T(".prompts:output_format.component").r(),
                hypothesis_specification=T(".prompts:hypothesis_specification").r(sota_solution=sota_solution),
            )

            component_user_prompt = T(".prompts:hypothesis_gen.user").r(
                targets="data science project",
                hypothesis_and_feedback=hypothesis_and_feedback,
            )

            resp_dict_component: dict = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    component_user_prompt, component_sys_prompt, json_mode=True
                )
            )

            component = resp_dict_component.get("component", "Component not provided")

            # Why we should split component selection and hpothesis generation
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
                    hypothesis_and_feedback=hypothesis_and_feedback,
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
                score_df = pd.read_csv(sota_exp.experiment_workspace.workspace_path / "score.csv", index_col=0)
                metric_name = score_df.columns[0]
                for fname in sota_exp.experiment_workspace.file_dict:
                    if re.match(r"^model_.+\.py", fname):
                        model_str = f"{fname}:\n{metric_name} on valid: {score_df.loc[fname[:-3]]}\n```python\n{sota_exp.experiment_workspace.file_dict[fname]}\n```\n"
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
                    hypothesis_and_feedback=hypothesis_and_feedback,
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
                    spec=sota_exp.experiment_workspace.file_dict["spec/data_loader.md"],
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.data_loader").r(),
                    hypothesis_and_feedback=hypothesis_and_feedback,
                )

                dt = DataLoaderTask(
                    name="Data loader and specification generation",
                    description=resp_dict.get(
                        "description", "Data loader and specification generation description not provided"
                    ),
                )

                exp = DSExperiment(sub_tasks=[dt], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
                return exp
            elif hypothesis.component == "FeatureEng":
                # TODO: RAG
                resp_dict = self.llm_task_gen(
                    targets="Feature Engineering",
                    scenario_desc=scenario_desc,
                    spec=sota_exp.experiment_workspace.file_dict["spec/feature.md"],
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.feature").r(),
                    hypothesis_and_feedback=hypothesis_and_feedback,
                )

                ft = FeatureTask(
                    name="Feature Engineering",
                    description=resp_dict.get("description", "Feature description not provided"),
                )

                exp = DSExperiment(sub_tasks=[ft], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
                return exp
            elif hypothesis.component == "Model":
                resp_dict = self.llm_task_gen(
                    scenario_desc=scenario_desc,
                    spec=sota_exp.experiment_workspace.file_dict["spec/model.md"],
                    hypothesis=hypothesis,
                    workspace_code=sota_exp.experiment_workspace.all_codes,
                    task_output_format=T(".prompts:output_format.model").r(),
                    hypothesis_and_feedback=hypothesis_and_feedback,
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
                exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
                return exp
            elif hypothesis.component == "Ensemble":
                resp_dict = self.llm_task_gen(
                    targets="Ensemble",
                    scenario_desc=scenario_desc,
                    spec=sota_exp.experiment_workspace.file_dict["spec/ensemble.md"],
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.ensemble").r(),
                    hypothesis_and_feedback=hypothesis_and_feedback,
                )

                et = EnsembleTask(
                    name="Ensemble",
                    description=resp_dict.get("description", "Ensemble description not provided"),
                )

                exp = DSExperiment(sub_tasks=[et], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
                return exp
            elif hypothesis.component == "Workflow":
                resp_dict = self.llm_task_gen(
                    targets="Workflow",
                    scenario_desc=scenario_desc,
                    spec=sota_exp.experiment_workspace.file_dict["spec/workflow.md"],
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.workflow").r(),
                    hypothesis_and_feedback=hypothesis_and_feedback,
                )

                wt = WorkflowTask(
                    name="Workflow",
                    description=resp_dict.get("description", "Workflow description not provided"),
                )

                exp = DSExperiment(sub_tasks=[wt], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(sota_exp.experiment_workspace.workspace_path)
                return exp

        return super().gen(trace)
