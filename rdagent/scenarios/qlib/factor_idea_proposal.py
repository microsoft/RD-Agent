import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from rdagent.components.idea_proposal.factor_proposal import (
    FactorHypothesis,
    FactorHypothesisGen,
)
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Scenario, Trace

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

QlibFactorHypothesis = FactorHypothesis


class QlibFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.gen_json_flag = True

    def prepare_gen_context(self, trace: Trace) -> None:
        self.gen_context_flag = True

        hypothesis_feedback = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=trace)
        )
        self.gen_context_dict = {
            "hypothesis_and_feedback": hypothesis_feedback,
            "RAG": ...,
            "factor_output_format": prompt_dict["output_format"],
        }

    def gen_response_to_hypothesis_list(self, response: str) -> FactorHypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibFactorHypothesis(hypothesis=response_dict["hypothesis"], reason=response_dict["reason"])
        return hypothesis
