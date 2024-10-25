from abc import abstractmethod
from pathlib import Path
from typing import Tuple

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.model_coder.model import ModelExperiment
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import (
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Scenario,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend

ModelHypothesis = Hypothesis

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class ModelHypothesisGen(HypothesisGen):
    prompts: Prompts = prompt_dict

    # The following methods are scenario related so they should be implemented in the subclass
    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str) -> ModelHypothesis: ...

    def gen(self, trace: Trace) -> ModelHypothesis:
        context_dict, json_flag = self.prepare_context(trace)

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(ModelHypothesisGen.prompts["hypothesis_gen"]["system_prompt"])
            .render(
                targets="model tuning",
                scenario=self.scen.get_scenario_all_desc(),
                hypothesis_output_format=context_dict["hypothesis_output_format"],
                hypothesis_specification=context_dict["hypothesis_specification"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(ModelHypothesisGen.prompts["hypothesis_gen"]["user_prompt"])
            .render(
                targets="model tuning",
                RAG=context_dict["RAG"],
            )
        )

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)

        hypothesis = self.convert_response(resp)

        return hypothesis


class ModelHypothesis2Experiment(Hypothesis2Experiment[ModelExperiment]):
    prompts: Prompts = prompt_dict

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, trace: Trace) -> ModelExperiment: ...

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> ModelExperiment:
        context, json_flag = self.prepare_context(hypothesis, trace)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(ModelHypothesis2Experiment.prompts["hypothesis2experiment"]["system_prompt"])
            .render(
                targets="feature engineering and model building",
                scenario=trace.scen.get_scenario_all_desc(),
                experiment_output_format=context["experiment_output_format"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(ModelHypothesis2Experiment.prompts["hypothesis2experiment"]["user_prompt"])
            .render(
                targets="feature engineering and model building",
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                target_list=context["target_list"],
                RAG=context["RAG"],
            )
        )

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)

        return self.convert_response(resp, trace)
