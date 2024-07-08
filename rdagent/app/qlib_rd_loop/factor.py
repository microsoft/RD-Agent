"""
TODO: Factor Structure RD-Loop
"""
from dotenv import load_dotenv
import pandas as pd
from typing import Optional
 
load_dotenv(override=True)
# import_from
from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.core.proposal import (
    HypothesisExperiment2Feedback,
    Hypothesis2Experiment,
    HypothesisGen,
    HypothesisSet,
    Trace,
)
from rdagent.core.task_generator import TaskGenerator
from rdagent.core.utils import import_class

#TODO: Adjust the placement
class ExperimentResults:
    def __init__(self, last_experiment: Optional[pd.DataFrame] = None, sota: Optional[pd.DataFrame] = None, alpha158: Optional[pd.DataFrame] = None):
        self.last_experiment = last_experiment
        self.sota = sota
        self.alpha158 = alpha158

    def update_last_experiment(self, new_result: pd.DataFrame):
        self.last_experiment = new_result

    def update_sota(self, new_sota: pd.DataFrame):
        self.sota = new_sota

    def update_alpha158(self, new_alpha158: pd.DataFrame):
        self.alpha158 = new_alpha158

scen = import_class(PROP_SETTING.scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()

qlib_factor_coder: TaskGenerator = import_class(PROP_SETTING.qlib_factor_coder)()
qlib_factor_runner: TaskGenerator = import_class(PROP_SETTING.qlib_factor_runner)()

qlib_factor_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.qlib_factor_summarizer)()


trace = Trace(scen=scen)
hs = HypothesisSet(trace=trace)
hypothesis = hypothesis_gen.gen(trace)
exp_res_all = ExperimentResults()
#TODO: Add alpha158 result (also SOTA)

# for _ in range(PROP_SETTING.evolving_n):
#     hypothesis = hypothesis_gen.gen(trace)
#     exp = hypothesis2experiment.convert(hs)
    
# Get factor data
import pickle
file_path = '/home/finco/RDAgent_MS/RD-Agent/git_ignore_folder/factor_data_output/exp.pkl'
with open(file_path, 'rb') as file:
    exp = pickle.load(file)

exp = qlib_factor_runner.generate(exp)
#TODO: Just a test
exp_res_all.update_last_experiment(exp.result)
exp_res_all.update_sota(exp.result)
exp_res_all.update_alpha158(exp.result)
print(exp.result)

feedback = qlib_factor_summarizer.generateFeedback(exp, hypothesis, exp_res_all)
if feedback.replace_sota:
    exp_res_all.update_sota(exp.result)
exp_res_all.update_last_experiment(exp.result)
# trace.hist.append((hypothesis, exp, feedback))
