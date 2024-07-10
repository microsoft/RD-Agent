import json
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask, ModelResult
from rdagent.components.proposal.model_proposal import (
    ModelHypothesis,
    ModelHypothesis2Experiment,
    ModelHypothesisGen,
)
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import HypothesisSet, Scenario, Trace

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

QlibModelHypothesis = ModelHypothesis


class QlibModelHypothesisGen(ModelHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_feedback = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=trace)
        )
        context_dict = {
            "hypothesis_and_feedback": hypothesis_feedback,
            "RAG": ...,
            "hypothesis_output_format": prompt_dict["hypothesis_output_format"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> ModelHypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibModelHypothesis(hypothesis=response_dict["hypothesis"], reason=response_dict["reason"])
        return hypothesis


class QlibModelHypothesis2Experiment(ModelHypothesis2Experiment):
    def prepare_context(self, hs: HypothesisSet) -> Tuple[dict, bool]:
        scenario = hs.trace.scen.get_scenario_all_desc()
        experiment_output_format = prompt_dict["experiment_output_format"]

        hypothesis_and_feedback = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=hs.trace)
        )

        experiment_list: List[ModelExperiment] = [t[1] for t in hs.trace.hist]

        model_list = []
        for experiment in experiment_list:
            model_list.extend(experiment.sub_tasks)

        return {
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "model_list": model_list,
            "RAG": ...,
        }, True

    def convert_response(self, response: str) -> ModelExperiment:
        response_dict = json.loads(response)
        tasks = []
        for model_name in response_dict:
            description = response_dict[model_name]["description"]
            architecture = response_dict[model_name]["architecture"]
            hyperparameters = response_dict[model_name]["hyperparameters"]
            tasks.append(ModelTask(model_name, description, architecture, hyperparameters))
        return ModelExperiment(tasks)
