import json
import math
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined
from rdagent.core.prompts import Prompts


from rdagent.core.proposal import Hypothesis, Scenario, Trace
import random
from rdagent.components.proposal import (
    FactorAndModelHypothesis2Experiment,
    FactorAndModelHypothesisGen,
)

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")

class QlibQuantHypothesis(Hypothesis):
    def __init__(
        self,
        hypothesis: str,
        reason: str,
        concise_reason: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        action: str,
    ) -> None:
        super().__init__(
            hypothesis, reason, concise_reason, concise_observation, concise_justification, concise_knowledge
        )
        self.action = action

    def __str__(self) -> str:
        return f"""Chosen Action: {self.action}
Hypothesis: {self.hypothesis}
Reason: {self.reason}
"""
    
class QlibQuantHypothesisGen(FactorAndModelHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        action = random.choice(["factor", "model"])
        self.targets = action

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": "In Quantitative Finance, market data could be time-series, and GRU model/LSTM model are suitable for them. Do not generate GNN model as for now." if action == "model" else None,
            "hypothesis_output_format": prompt_dict["hypothesis_output_format_with_action"],
            "hypothesis_specification": prompt_dict["factor_hypothesis_specification"] if action == "factor" else prompt_dict["model_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibQuantHypothesis(
            hypothesis=response_dict["hypothesis"],
            reason=response_dict["reason"],
            concise_reason=response_dict["concise_reason"],
            concise_observation=response_dict["concise_observation"],
            concise_justification=response_dict["concise_justification"],
            concise_knowledge=response_dict["concise_knowledge"],
            action=response_dict["action"],
        )
        return hypothesis
    
