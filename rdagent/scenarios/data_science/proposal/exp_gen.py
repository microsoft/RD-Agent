from argparse import ONE_OR_MORE
from typing import Literal
import json

from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask

from rdagent.scenarios.data_science.experiment.experiment import DataLoaderExperiment, FeatureExperiment, ModelExperiment, EnsembleExperiment, WorkflowExperiment

from rdagent.components.proposal import LLMHypothesis2Experiment, LLMHypothesisGen
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


class DSHypothesisGen(LLMHypothesisGen):
    def get_next_action(self, trace):
        pass

    def prepare_context(self, trace):
        hypothesis_and_feedback = T(".prompts:hypothesis_and_feedback").r(trace=trace)
        
        # TODO: how to generate sota solution
        sota_solution = ""
        hypothesis_specification = T(".prompts:hypothesis_specification").r(sota_solution=sota_solution)
        
        return {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            # TODO: "RAG": "",
            "hypothesis_output_format": T(".prompts:output_format.hypothesis").r(),
            "hypothesis_specification": hypothesis_specification,
        }, True

    def convert_response(self, response):
        response_dict = json.loads(response)
        return DSHypothesis(
            hypothesis=response_dict.get("hypothesis", "Hypothesis not provided"),
            reason=response_dict.get("reason", "Reason not provided"),
            concise_reason=response_dict.get("concise_reason", "Concise reason not provided"),
            concise_observation=response_dict.get("concise_observation", "Concise observation not provided"),
            concise_justification=response_dict.get("concise_justification", "Concise justification not provided"),
            concise_knowledge=response_dict.get("concise_knowledge", "Concise knowledge not provided"),
            component=response_dict.get("component", "Component not provided"),
        )


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
            # proposal + design
            hypothesis: DSHypothesis = DSHypothesisGen(scen=self.scen).gen(trace)
            scenario = trace.scen.get_scenario_all_desc()
            
            if hypothesis.component == "DataLoadSpec":
                pass
            elif hypothesis.component == "FeatureEng":
                pass
            elif hypothesis.component == "Model":
                pass
            elif hypothesis.component == "Ensemble":
                pass
            elif hypothesis.component == "Workflow":
                pass
        else:
            for o in ORDER:
                if o in successful_components:
                    # we already have the component, then skip
                    continue
                elif o == "DataLoadSpec":
                    dlt = DataLoaderTask(name="DataLoaderTask", description="")
                    exp = DataLoaderExperiment(
                        sub_tasks=[dlt],
                    )
                    self.complete_component.add(o)
                    return exp
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
