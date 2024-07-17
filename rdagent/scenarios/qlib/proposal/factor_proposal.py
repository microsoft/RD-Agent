import json
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.proposal.factor_proposal import (
    FactorHypothesis,
    FactorHypothesis2Experiment,
    FactorHypothesisGen,
)
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")

QlibFactorHypothesis = FactorHypothesis


class QlibFactorHypothesisGen(FactorHypothesisGen):
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

    def convert_response(self, response: str) -> FactorHypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibFactorHypothesis(hypothesis=response_dict["hypothesis"], reason=response_dict["reason"])
        return hypothesis


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = prompt_dict["factor_experiment_output_format"]

        hypothesis_and_feedback = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=trace)
        )

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "RAG": ...,
        }, True

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []
        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            tasks.append(FactorTask(factor_name, description, formulation, variables))
        exp = QlibFactorExperiment(tasks)
        exp.based_experiments = [t[1] for t in trace.hist if t[2]]
        if len(exp.based_experiments) == 0:
            exp.based_experiments.append(QlibFactorExperiment(sub_tasks=[]))
        return exp
