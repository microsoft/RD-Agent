import json
import random
from typing import Tuple

from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
from rdagent.components.proposal import FactorAndModelHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.proposal.bandit import (
    EnvController,
    extract_metrics_from_experiment,
)
from rdagent.utils.agent.tpl import T


class QuantTrace(Trace):
    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        # Initialize the controller with default weights
        self.controller = EnvController()


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

        # ========= Bandit ==========
        if QUANT_PROP_SETTING.action_selection == "bandit":
            if len(trace.hist) > 0:
                metric = extract_metrics_from_experiment(trace.hist[-1][0])
                prev_action = trace.hist[-1][0].hypothesis.action
                trace.controller.record(metric, prev_action)
                action = trace.controller.decide(metric)
            else:
                action = "factor"
        # ========= LLM ==========
        elif QUANT_PROP_SETTING.action_selection == "llm":
            hypothesis_and_feedback = (
                T("scenarios.qlib.prompts:hypothesis_and_feedback").render(trace=trace)
                if len(trace.hist) > 0
                else "No previous hypothesis and feedback available since it's the first round."
            )

            last_hypothesis_and_feedback = (
                T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                    experiment=trace.hist[-1][0], feedback=trace.hist[-1][1]
                )
                if len(trace.hist) > 0
                else "No previous hypothesis and feedback available since it's the first round."
            )

            system_prompt = T("scenarios.qlib.prompts:action_gen.system").r()
            user_prompt = T("scenarios.qlib.prompts:action_gen.user").r(
                hypothesis_and_feedback=hypothesis_and_feedback,
                last_hypothesis_and_feedback=last_hypothesis_and_feedback,
            )
            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)

            action = json.loads(resp).get("action", "factor")
        # ========= random ==========
        elif QUANT_PROP_SETTING.action_selection == "random":
            action = random.choice(["factor", "model"])
        self.targets = action

        qaunt_rag = None
        if action == "factor":
            if len(trace.hist) < 6:
                qaunt_rag = "Try the easiest and fastest factors to experiment with from various perspectives first."
            else:
                qaunt_rag = "Now, you need to try factors that can achieve high IC (e.g., machine learning-based factors)! Do not include factors that are similar to those in the SOTA factor library!"
        elif action == "model":
            qaunt_rag = "1. In Quantitative Finance, market data could be time-series, and GRU model/LSTM model are suitable for them. Do not generate GNN model as for now.\n2. The training data consists of approximately 478,000 samples for the training set and about 128,000 samples for the validation set. Please design the hyperparameters accordingly and control the model size. This has a significant impact on the training results. If you believe that the previous model itself is good but the training hyperparameters or model hyperparameters are not optimal, you can return the same model and adjust these parameters instead.\n"

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
        else:
            specific_trace = Trace(trace.scen)
            if action == "factor":
                # all factor experiments and the SOTA model experiment
                model_inserted = False
                for i in range(len(trace.hist) - 1, -1, -1):  # Reverse iteration
                    if trace.hist[i][0].hypothesis.action == "factor":
                        specific_trace.hist.insert(0, trace.hist[i])
                    elif (
                        trace.hist[i][0].hypothesis.action == "model"
                        and trace.hist[i][1].decision is True
                        and model_inserted == False
                    ):
                        specific_trace.hist.insert(0, trace.hist[i])
                        model_inserted = True
            elif action == "model":
                # all model experiments and all SOTA factor experiments
                factor_inserted = False
                for i in range(len(trace.hist) - 1, -1, -1):  # Reverse iteration
                    if trace.hist[i][0].hypothesis.action == "model":
                        specific_trace.hist.insert(0, trace.hist[i])
                    elif (
                        trace.hist[i][0].hypothesis.action == "factor"
                        and trace.hist[i][1].decision is True
                        and factor_inserted == False
                    ):
                        specific_trace.hist.insert(0, trace.hist[i])
                        factor_inserted = True
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                    trace=specific_trace,
                )
            else:
                hypothesis_and_feedback = "No previous hypothesis and feedback available."

        last_hypothesis_and_feedback = None
        for i in range(len(trace.hist) - 1, -1, -1):
            if trace.hist[i][0].hypothesis.action == action:
                last_hypothesis_and_feedback = T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                    experiment=trace.hist[i][0], feedback=trace.hist[i][1]
                )
                break

        sota_hypothesis_and_feedback = None
        if action == "model":
            for i in range(len(trace.hist) - 1, -1, -1):
                if trace.hist[i][0].hypothesis.action == "model" and trace.hist[i][1].decision is True:
                    sota_hypothesis_and_feedback = T("scenarios.qlib.prompts:sota_hypothesis_and_feedback").r(
                        experiment=trace.hist[i][0], feedback=trace.hist[i][1]
                    )
                    break

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "SOTA_hypothesis_and_feedback": sota_hypothesis_and_feedback,
            "RAG": qaunt_rag,
            "hypothesis_output_format": T("scenarios.qlib.prompts:hypothesis_output_format_with_action").r(),
            "hypothesis_specification": (
                T("scenarios.qlib.prompts:factor_hypothesis_specification").r()
                if action == "factor"
                else T("scenarios.qlib.prompts:model_hypothesis_specification").r()
            ),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibQuantHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
            action=response_dict.get("action"),
        )
        return hypothesis
