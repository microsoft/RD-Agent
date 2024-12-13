from argparse import ONE_OR_MORE
from typing import Literal
import json

from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask

from rdagent.scenarios.data_science.experiment.experiment import DataLoaderExperiment, FeatureExperiment, ModelExperiment, EnsembleExperiment, WorkflowExperiment

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import ExpGen, Trace, Hypothesis
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
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

class DSExpGen(ExpGen):
    """Data Science Task Generator."""

    def gen(self, trace: Trace) -> Experiment:
        successful_components = set()
        for h, _, hf in trace.hist:
            if hf.decision:
                successful_components.add(h.component)
        
        def is_complete():
            """is all components complete"""
            return set(ORDER) == successful_components

        if is_complete():
            # base info
            scenario = trace.scen.get_scenario_all_desc()
            hypothesis_and_feedback = T(".prompts:hypothesis_and_feedback").r(trace=trace)
            
            # 1. hypothesis gen
            # TODO: how to generate sota solution
            sota_solution = ""
            system_prompt = T(".prompts:hypothesis_gen.system").r(
                targets="data science project",
                scenario=scenario,
                hypothesis_output_format=T(".prompts:output_format.hypothesis").r(),
                hypothesis_specification=T(".prompts:hypothesis_specification").r(sota_solution=sota_solution),
                )
            user_prompt = T(".prompts:hypothesis_gen.user").r(
                targets="data science project",
                hypothesis_and_feedback=hypothesis_and_feedback,
                )

            resp_dict = json.loads(APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True))
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
                data_loader_task_output_format = T(".prompts:output_format.data_loader").r()
                system_prompt = T(".prompts:task_gen.system").r(
                    targets="Data loader and specification generation",
                    scenario=scenario,
                    hypothesis=hypothesis,
                    task_output_format=data_loader_task_output_format,
                    )
                user_prompt = T(".prompts:task_gen.user").r(
                    targets="Data loader and specification generation",
                    hypothesis=hypothesis,
                    hypothesis_and_feedback=hypothesis_and_feedback,
                    )
                
                resp_dict = json.loads(APIBackend().build_messages_and_create_chat_completion(user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True))
                dt = DataLoaderTask(
                    name="Data loader and specification generation",
                    description=resp_dict.get("description", "Data loader and specification generation description not provided"),
                )
                
                return DataLoaderExperiment(sub_tasks=[dt], hypothesis=hypothesis)
            elif hypothesis.component == "FeatureEng":
                # TODO: RAG
                feature_task_output_format = T(".prompts:output_format.feature").r()
                system_prompt = T(".prompts:task_gen.system").r(
                    targets="Feature Engineering",
                    scenario=scenario,
                    hypothesis=hypothesis,
                    task_output_format=feature_task_output_format,
                    )
                user_prompt = T(".prompts:task_gen.user").r(
                    targets="Feature Engineering",
                    hypothesis=hypothesis,
                    hypothesis_and_feedback=hypothesis_and_feedback,
                    )
                
                resp_dict = json.loads(APIBackend().build_messages_and_create_chat_completion(user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True))
                tasks = []
                for fn in resp_dict:
                    ft = FeatureTask(
                        name=fn,
                        description=resp_dict[fn].get("description", "Factor description not provided"),
                        formulation=resp_dict[fn].get("formulation", "Feature formulation not provided"),
                        variables=resp_dict[fn].get("variables", "Variables not provided"),
                        )
                
                return FeatureExperiment(sub_tasks=tasks, hypothesis=hypothesis)
            elif hypothesis.component == "Model":
                model_task_output_format = T(".prompts:output_format.model").r()
                
                system_prompt = T(".prompts:task_gen.system").r(
                    targets="Models",
                    scenario=scenario,
                    hypothesis=hypothesis,
                    task_output_format=model_task_output_format,
                    )
                user_prompt = T(".prompts:task_gen.user").r(
                    targets="Models",
                    hypothesis=hypothesis,
                    hypothesis_and_feedback=hypothesis_and_feedback,
                    )
                
                resp_dict = json.loads(APIBackend().build_messages_and_create_chat_completion(user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True))
                mt = ModelTask(
                    name=resp_dict.get("model_name", "Model name not provided"),
                    description=resp_dict.get("description", "Model description not provided"),
                    architecture=resp_dict.get("architecture", "Model architecture not provided"),
                    hyperparameters=resp_dict.get("hyperparameters", "Model hyperparameters not provided"),
                    base_code="",
                )
                
                return ModelExperiment(sub_tasks=[mt], hypothesis=hypothesis)
            elif hypothesis.component == "Ensemble":
                ensemble_task_output_format = T(".prompts:output_format.ensemble").r()
                
                system_prompt = T(".prompts:task_gen.system").r(
                    targets="Ensemble",
                    scenario=scenario,
                    hypothesis=hypothesis,
                    task_output_format=ensemble_task_output_format,
                    )
                user_prompt = T(".prompts:task_gen.user").r(
                    targets="Ensemble",
                    hypothesis=hypothesis,
                    hypothesis_and_feedback=hypothesis_and_feedback,
                    )
                
                resp_dict = json.loads(APIBackend().build_messages_and_create_chat_completion(user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True))
                et = EnsembleTask(
                    name="Ensemble",
                    description=resp_dict.get("description", "Ensemble description not provided"),
                )

                return EnsembleExperiment(sub_tasks=[et], hypothesis=hypothesis)                
            elif hypothesis.component == "Workflow":
                workflow_task_output_format = T(".prompts:output_format.workflow").r()
                
                system_prompt = T(".prompts:task_gen.system").r(
                    targets="Workflow",
                    scenario=scenario,
                    hypothesis=hypothesis,
                    task_output_format=workflow_task_output_format,
                    )
                user_prompt = T(".prompts:task_gen.user").r(
                    targets="Workflow",
                    hypothesis=hypothesis,
                    hypothesis_and_feedback=hypothesis_and_feedback,
                    )
                
                resp_dict = json.loads(APIBackend().build_messages_and_create_chat_completion(user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True))
                wt = WorkflowTask(
                    name="Workflow",
                    description=resp_dict.get("description", "Workflow description not provided"),
                )

                return WorkflowExperiment(sub_tasks=[wt], hypothesis=hypothesis)
        else:
            for o in ORDER:
                if o in successful_components:
                    # we already have the component, then skip
                    continue
                elif o == "DataLoadSpec":
                    data_loader_task_output_format = T(".prompts:output_format.data_loader").r()
                    system_prompt = T(".prompts:task_gen.system").r(
                        targets="Data loader and specification generation",
                        scenario=scenario,
                        hypothesis=None,
                        task_output_format=data_loader_task_output_format,
                    )
                    user_prompt = T(".prompts:task_gen.user").r(
                        targets="Data loader and specification generation",
                        hypothesis=None,
                    )
                    
                    resp_dict = json.loads(APIBackend().build_messages_and_create_chat_completion(user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True))
                    dt = DataLoaderTask(
                        name="Data loader and specification generation",
                        description=resp_dict.get("description", "Data loader and specification generation description not provided"),
                    )
                
                    return DataLoaderExperiment(sub_tasks=[dt], hypothesis=hypothesis)
                elif o == "FeatureEng":
                    ft = FeatureTask(name="FeatureTask", description="")
                    exp = FeatureExperiment(
                        sub_tasks=[ft],
                    )
                    self.complete_component.add(o)
                    return exp
                elif o == "Model":
                    mt = ModelTask(name="ModelTask", description="")
                    exp = ModelExperiment(
                        sub_tasks=[mt],
                    )
                    self.complete_component.add(o)
                    return exp
                elif o == "Ensemble":
                    et = EnsembleTask(name="EnsembleTask", description="")
                    exp = EnsembleExperiment(
                        sub_tasks=[et],
                    )
                    self.complete_component.add(o)
                    return exp
                elif o == "Workflow":
                    wt = WorkflowTask(name="WorkflowTask", description="")
                    exp = WorkflowExperiment(
                        sub_tasks=[wt],
                    )
                    self.complete_component.add(o)
                    return exp
        return super().gen(trace)
