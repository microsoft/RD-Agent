import json
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.factor import FactorExperiment, FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")

QlibFactorHypothesis = Hypothesis


class QlibFactorHypothesisGen(FactorHypothesisGen):
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

        last_hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["last_hypothesis_and_feedback"])
                .render(experiment=trace.hist[-1][0],
                        feedback=trace.hist[-1][1])
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "RAG": "Try the easiest and fastest factors to experiment with from various perspectives first. Also, try to use fundamental factors as much as possible." if len(trace.hist) < 10 else """Now, you need to try factors that can achieve high IC (e.g., machine learning-based factors). The following factor is a good example, and you MUST aim for such factors. At the same time, try to use fundamental factors as much as possible.
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor


    def calculate_volatility_sentiment_50d_dynamicadjustments():
        daily_f = pd.read_hdf('daily_f.h5', key='data')
        daily_pv = pd.read_hdf('daily_pv.h5', key='data')
        daily_pv.sort_values(by=['datetime', 'instrument'], inplace=True)

        daily_pv['returns'] = daily_pv.groupby('instrument')['$close'].pct_change()
        daily_pv['volatility_50d'] = daily_pv.groupby('instrument')['returns'].transform(lambda x: x.rolling(window=50).std())

        # Placeholder for actual sentiment data integration
        # Replace the following line with actual sentiment data retrieval and integration
        daily_pv['sentiment'] = np.random.uniform(-1, 1, size=len(daily_pv))  # This should be replaced

        def get_dynamic_adjustments(df):
        model = GradientBoostingRegressor()
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['returns'].fillna(0).values
        model.fit(X, y)
        return pd.Series(model.predict(X), index=df.index)

        daily_pv['dynamic_adjustments'] = daily_pv.groupby('instrument').apply(get_dynamic_adjustments).reset_index(level=0, drop=True)

        daily_pv['Volatility_Sentiment_50D_DynamicAdjustments'] = (
        daily_pv['volatility_50d'] * daily_pv['sentiment'] * daily_pv['dynamic_adjustments']
        )

        result = daily_pv[['Volatility_Sentiment_50D_DynamicAdjustments']].dropna()
        result.to_hdf('result.h5', key='data', mode='w')


    if __name__ == '__main__':
        calculate_volatility_sentiment_50d_dynamicadjustments()""",
            "hypothesis_output_format": prompt_dict["factor_hypothesis_output_format"],
            "hypothesis_specification": prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = json.loads(response)
        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict["hypothesis"],
            reason=response_dict["reason"],
            concise_reason=response_dict["concise_reason"],
            concise_observation=response_dict["concise_observation"],
            concise_justification=response_dict["concise_justification"],
            concise_knowledge=response_dict["concise_knowledge"],
        )
        return hypothesis


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = prompt_dict["factor_experiment_output_format"]

        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        experiment_list: List[FactorExperiment] = [t[0] for t in trace.hist]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
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
                )
            )

        exp = QlibFactorExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[0] for t in trace.hist if t[1]]

        unique_tasks = []

        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
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
