import json

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.utils.agent.tpl import T

QlibFactorHypothesis = Hypothesis


def _generate_dynamic_rag(trace: Trace) -> str:
    """Generate RAG advice based on exploration history."""
    direction_keywords = {
        "momentum": ["momentum", "roc", "return", "price change"],
        "volatility": ["volatility", "std", "variance", "risk"],
        "volume": ["volume", "vwap", "turnover", "flow"],
        "correlation": ["correlation", "corr", "cord"],
        "distribution": ["kurtosis", "skewness", "zscore", "distribution"],
        "ml": ["machine learning", "neural", "transform", "predict"],
        "liquidity": ["liquidity", "bid", "ask", "spread"],
        "sentiment": ["sentiment", "news", "analyst", "emotion"],
        "tail_risk": ["tail", "extreme", "var", "drawdown"],
    }

    if len(trace.hist) == 0:
        return (
            "Start with high-quality factors from underexplored directions: "
            "liquidity risk, order flow imbalances, sentiment proxies, tail risk measures. "
            "Avoid duplicating existing baseline factors."
        )

    explored = set()
    for exp, _ in trace.hist:
        if hasattr(exp, "hypothesis") and hasattr(exp.hypothesis, "hypothesis"):
            hypothesis_text = exp.hypothesis.hypothesis.lower()
            for direction, keywords in direction_keywords.items():
                if any(kw in hypothesis_text for kw in keywords):
                    explored.add(direction)

    underexplored = ["liquidity", "sentiment", "tail_risk", "order_flow"]
    recommendations = [d for d in underexplored if d not in explored]

    if len(trace.hist) < 5:
        return (
            "Focus on simple, interpretable factors. "
            "Start with underexplored directions: liquidity risk, order flow imbalances."
        )

    if recommendations:
        return (
            f"Focus on underexplored directions: {', '.join(recommendations[:3])}. "
            "These have high potential for alpha generation."
        )

    return (
        "Consider machine learning-based factors or enhancing existing directions "
        "with better normalization techniques."
    )


class QlibFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> tuple[dict, bool]:
        hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                trace=trace,
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        last_hypothesis_and_feedback = (
            T("scenarios.qlib.prompts:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0], feedback=trace.hist[-1][1],
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "baseline_context": T("scenarios.qlib.prompts:baseline_context").r(),
            "RAG": _generate_dynamic_rag(trace),
            "hypothesis_output_format": T("scenarios.qlib.prompts:factor_hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.qlib.prompts:factor_hypothesis_specification").r(),
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict.get("hypothesis"),
            reason=response_dict.get("reason"),
            concise_reason=response_dict.get("concise_reason"),
            concise_observation=response_dict.get("concise_observation"),
            concise_justification=response_dict.get("concise_justification"),
            concise_knowledge=response_dict.get("concise_knowledge"),
        )
        return hypothesis


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> tuple[dict | bool]:
        if isinstance(trace.scen, QlibQuantScenario):
            scenario = trace.scen.get_scenario_all_desc(action="factor")
        else:
            scenario = trace.scen.get_scenario_all_desc()

        experiment_output_format = T("scenarios.qlib.prompts:factor_experiment_output_format").r()

        if len(trace.hist) == 0:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round."
        else:
            specific_trace = Trace(trace.scen)
            for i in range(len(trace.hist) - 1, -1, -1):
                if not hasattr(trace.hist[i][0].hypothesis, "action") or trace.hist[i][0].hypothesis.action == "factor":
                    specific_trace.hist.insert(0, trace.hist[i])
            if len(specific_trace.hist) > 0:
                specific_trace.hist.reverse()
                hypothesis_and_feedback = T("scenarios.qlib.prompts:hypothesis_and_feedback").r(
                    trace=specific_trace,
                )
            else:
                hypothesis_and_feedback = "No previous hypothesis and feedback available."

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": [],
            "RAG": None,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        response_dict = json.loads(response)
        tasks = []

        for factor_name in response_dict:
            description = response_dict[factor_name]["description"]
            formulation = response_dict[factor_name]["formulation"]
            variables = response_dict[factor_name]["variables"]
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    variables=variables,
                ),
            )

        exp = QlibFactorExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [
            t[0] for t in trace.hist if t[1] and isinstance(t[0], FactorExperiment)
        ]

        unique_tasks = []
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                if isinstance(based_exp, QlibModelExperiment):
                    continue
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp
