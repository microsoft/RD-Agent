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
from rdagent.scenarios.qlib.task_generator.data import ExperimentResults


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
exp_res_all.setup()

# for _ in range(PROP_SETTING.evolving_n):
#     hypothesis = hypothesis_gen.gen(trace)
#     exp = hypothesis2experiment.convert(hs)
    
# Get factor data for test
import pickle
file_path = '/home/finco/RDAgent_MS/RD-Agent/git_ignore_folder/factor_data_output/exp.pkl'
with open(file_path, 'rb') as file:
    exp = pickle.load(file)

exp = qlib_factor_runner.generate(exp)

feedback = qlib_factor_summarizer.generateFeedback(exp, hypothesis, exp_res_all)
exp_res_all.update_last_experiment(exp.result)
# trace.hist.append((hypothesis, exp, feedback))
