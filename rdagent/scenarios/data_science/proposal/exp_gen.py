import json
from typing import Literal

from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.experiment import Experiment
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.core.proposal import ExpGen, Hypothesis, HypothesisFeedback, Trace
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.utils.agent.tpl import T

COMPONENT = Literal["DataLoadSpec", "FeatureEng", "Model", "Ensemble", "Workflow"]
ORDER = COMPONENT.__args__


class DSHypothesis(Hypothesis):
    def __init__(
        self,
        hypothesis: str,
        reason: str,
        concise_reason: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        component: COMPONENT,
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.component = component

    def __str__(self) -> str:
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
        self.hist: list[tuple[DSHypothesis, Experiment, HypothesisFeedback]] = []
        self.knowledge_base = knowledge_base

    def get_sota_hypothesis_and_experiment(
        self, component: COMPONENT | None = None
    ) -> tuple[DSHypothesis | None, Experiment | None]:
        """Access the last experiment result, sub-task, and the corresponding hypothesis."""
        for h, exp, hf in self.hist[::-1]:
            if hf.decision:
                if component and h.component != component:
                    continue
                return h, exp
        return None, None


class DSExpGen(ExpGen):
    """Data Science Task Generator."""

    def llm_task_gen(
        self,
        targets: str,
        scenario_desc: str,
        task_output_format: str,
        hypothesis: Hypothesis | None = None,
        hypothesis_and_feedback: str | None = None,
    ) -> dict:
        system_prompt = T(".prompts:task_gen.system").r(
            targets=targets,
            scenario=scenario_desc,
            hypothesis=hypothesis,
            task_output_format=task_output_format,
        )
        user_prompt = T(".prompts:task_gen.user").r(
            targets=targets,
            hypothesis=hypothesis,
            hypothesis_and_feedback=hypothesis_and_feedback,
        )

        resp_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
            )
        )

        return resp_dict

    def gen(self, trace: DSTrace) -> DSExperiment:
        successful_components = set()
        for h, _, hf in trace.hist:
            if hf.decision:
                successful_components.add(h.component)

        def is_complete():
            """is all components complete"""
            return set(ORDER) == successful_components

        scenario_desc = trace.scen.get_scenario_all_desc()
        if is_complete():
            # base info
            hypothesis_and_feedback = T(".prompts:hypothesis_and_feedback").r(trace=trace)

            # 1. hypothesis gen
            # TODO: how to generate sota solution
            sota_solution = ""
            system_prompt = T(".prompts:hypothesis_gen.system").r(
                targets="data science project",
                scenario=scenario_desc,
                hypothesis_output_format=T(".prompts:output_format.hypothesis").r(),
                hypothesis_specification=T(".prompts:hypothesis_specification").r(sota_solution=sota_solution),
            )
            user_prompt = T(".prompts:hypothesis_gen.user").r(
                targets="data science project",
                hypothesis_and_feedback=hypothesis_and_feedback,
            )

            resp_dict = json.loads(
                APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
            )
            hypothesis = DSHypothesis(
                hypothesis=resp_dict.get("hypothesis", "Hypothesis not provided"),
                reason=resp_dict.get("reason", "Reason not provided"),
                concise_reason=resp_dict.get("concise_reason", "Concise reason not provided"),
                concise_observation=resp_dict.get("concise_observation", "Concise observation not provided"),
                concise_justification=resp_dict.get("concise_justification", "Concise justification not provided"),
                concise_knowledge=resp_dict.get("concise_knowledge", "Concise knowledge not provided"),
                component=resp_dict.get("component", "Component not provided"),
            )

            # 2. gen experiment
            if hypothesis.component == "DataLoadSpec":
                resp_dict = self.llm_task_gen(
                    targets="Data loader and specification generation",
                    scenario_desc=scenario_desc,
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

                return DSExperiment(sub_tasks=[dt], hypothesis=hypothesis)
            elif hypothesis.component == "FeatureEng":
                # TODO: RAG
                resp_dict = self.llm_task_gen(
                    targets="Feature Engineering",
                    scenario_desc=scenario_desc,
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.feature").r(),
                    hypothesis_and_feedback=hypothesis_and_feedback,
                )

                dependency_exp = trace.get_sota_hypothesis_and_experiment("DataLoadSpec")
                tasks = []
                for fn in resp_dict:
                    ft = FeatureTask(
                        name=fn,
                        description=resp_dict[fn].get("description", "Factor description not provided"),
                        formulation=resp_dict[fn].get("formulation", "Feature formulation not provided"),
                        variables=resp_dict[fn].get("variables", "Variables not provided"),
                    )

                exp = DSExperiment(sub_tasks=tasks, hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(dependency_exp.experiment_workspace.workspace_path)
                return exp
            elif hypothesis.component == "Model":
                resp_dict = self.llm_task_gen(
                    targets="Models",
                    scenario_desc=scenario_desc,
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.model").r(),
                    hypothesis_and_feedback=hypothesis_and_feedback,
                )

                dependency_exp = trace.get_sota_hypothesis_and_experiment("FeatureEng")
                mt = ModelTask(
                    name=resp_dict.get("model_name", "Model name not provided"),
                    description=resp_dict.get("description", "Model description not provided"),
                    architecture=resp_dict.get("architecture", "Model architecture not provided"),
                    hyperparameters=resp_dict.get("hyperparameters", "Model hyperparameters not provided"),
                    base_code="",
                )

                exp = DSExperiment(sub_tasks=[mt], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(dependency_exp.experiment_workspace.workspace_path)
                return exp
            elif hypothesis.component == "Ensemble":
                resp_dict = self.llm_task_gen(
                    targets="Ensemble",
                    scenario_desc=scenario_desc,
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.ensemble").r(),
                    hypothesis_and_feedback=hypothesis_and_feedback,
                )

                dependency_exp = trace.get_sota_hypothesis_and_experiment("Model")
                et = EnsembleTask(
                    name="Ensemble",
                    description=resp_dict.get("description", "Ensemble description not provided"),
                )

                exp = DSExperiment(sub_tasks=[et], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(dependency_exp.experiment_workspace.workspace_path)
                return exp
            elif hypothesis.component == "Workflow":
                resp_dict = self.llm_task_gen(
                    targets="Workflow",
                    scenario_desc=scenario_desc,
                    hypothesis=hypothesis,
                    task_output_format=T(".prompts:output_format.workflow").r(),
                    hypothesis_and_feedback=hypothesis_and_feedback,
                )

                dependency_exp = trace.get_sota_hypothesis_and_experiment("Ensemble")
                wt = WorkflowTask(
                    name="Workflow",
                    description=resp_dict.get("description", "Workflow description not provided"),
                )

                exp = DSExperiment(sub_tasks=[wt], hypothesis=hypothesis)
                exp.experiment_workspace.inject_code_from_folder(dependency_exp.experiment_workspace.workspace_path)
                return exp
        else:
            for o in ORDER:
                if o in successful_components:
                    # we already have the component, then skip
                    continue
                elif o == "DataLoadSpec":
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

                    exp = DSExperiment(sub_tasks=[dt])
                    return exp
                elif o == "FeatureEng":
                    resp_dict = self.llm_task_gen(
                        targets="Feature Engineering",
                        scenario_desc=scenario_desc,
                        task_output_format=T(".prompts:output_format.feature").r(),
                    )
                    dependency_exp = trace.get_sota_hypothesis_and_experiment("DataLoadSpec")
                    tasks = []
                    for fn in resp_dict:
                        ft = FeatureTask(
                            name=fn,
                            description=resp_dict[fn].get("description", "Factor description not provided"),
                            formulation=resp_dict[fn].get("formulation", "Feature formulation not provided"),
                            variables=resp_dict[fn].get("variables", "Variables not provided"),
                        )
                        tasks.append(ft)
                    exp = DSExperiment(sub_tasks=tasks)
                    exp.experiment_workspace.inject_code_from_folder(dependency_exp.experiment_workspace.workspace_path)
                    return exp
                elif o == "Model":
                    resp_dict = self.llm_task_gen(
                        targets="Models",
                        scenario_desc=scenario_desc,
                        task_output_format=T(".prompts:output_format.model").r(),
                    )
                    dependency_exp = trace.get_sota_hypothesis_and_experiment("FeatureEng")
                    if last_model_exp := trace.get_sota_hypothesis_and_experiment("Model"):
                        # TODO: model only have one (named "model.py")?
                        base_code = last_model_exp.experiment_workspace.code_dict["model.py"]
                    else:
                        base_code = ""
                    mt = ModelTask(
                        name=resp_dict.get("model_name", "Model name not provided"),
                        description=resp_dict.get("description", "Model description not provided"),
                        architecture=resp_dict.get("architecture", "Model architecture not provided"),
                        hyperparameters=resp_dict.get("hyperparameters", "Model hyperparameters not provided"),
                        base_code=base_code,
                    )
                    exp = DSExperiment(sub_tasks=[mt])
                    exp.experiment_workspace.inject_code_from_folder(dependency_exp.experiment_workspace.workspace_path)
                    return exp
                elif o == "Ensemble":
                    resp_dict = self.llm_task_gen(
                        targets="Ensemble",
                        scenario_desc=scenario_desc,
                        task_output_format=T(".prompts:output_format.ensemble").r(),
                    )
                    dependency_exp = trace.get_sota_hypothesis_and_experiment("Model")
                    et = EnsembleTask(
                        name="Ensemble",
                        description=resp_dict.get("description", "Ensemble description not provided"),
                    )
                    exp = DSExperiment(sub_tasks=[et])
                    exp.experiment_workspace.inject_code_from_folder(dependency_exp.experiment_workspace.workspace_path)
                    return exp
                elif o == "Workflow":
                    resp_dict = self.llm_task_gen(
                        targets="Workflow",
                        scenario_desc=scenario_desc,
                        task_output_format=T(".prompts:output_format.workflow").r(),
                    )
                    dependency_exp = trace.get_sota_hypothesis_and_experiment("Ensemble")
                    wt = WorkflowTask(
                        name="Workflow",
                        description=resp_dict.get("description", "Workflow description not provided"),
                    )
                    exp = DSExperiment(sub_tasks=[wt])
                    exp.experiment_workspace.inject_code_from_folder(dependency_exp.experiment_workspace.workspace_path)
                    return exp

        return super().gen(trace)
