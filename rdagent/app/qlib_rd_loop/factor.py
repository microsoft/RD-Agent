"""
TODO: Factor Structure RD-Loop
"""

from dotenv import load_dotenv

from rdagent.core.scenario import Scenario

load_dotenv(override=True)

from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.task_generator import TaskGenerator
from rdagent.core.utils import import_class

scen: Scenario = import_class(PROP_SETTING.qlib_factor_scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.qlib_factor_hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.qlib_factor_hypothesis2experiment)()

qlib_factor_coder: TaskGenerator = import_class(PROP_SETTING.qlib_factor_coder)(scen)
qlib_factor_runner: TaskGenerator = import_class(PROP_SETTING.qlib_factor_runner)(scen)

qlib_factor_summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.qlib_factor_summarizer)(scen)


trace = Trace(scen=scen)
for _ in range(PROP_SETTING.evolving_n):
    hypothesis = hypothesis_gen.gen(trace)
    exp = hypothesis2experiment.convert(hypothesis, trace)
    exp = qlib_factor_coder.generate(exp)
    exp = qlib_factor_runner.generate(exp)
    feedback = qlib_factor_summarizer.generateFeedback(exp, hypothesis, trace)

    trace.hist.append((hypothesis, exp, feedback))


"""
trace = Trace(scen=scen)
# for _ in range(PROP_SETTING.evolving_n):
for _ in range(1):
    hypothesis = hypothesis_gen.gen(trace)
    exp = hypothesis2experiment.convert(hypothesis, trace)
    # exp = qlib_factor_coder.generate(exp)
    import pickle
    file_path = '/home/finco/v-yuanteli/RD-Agent/git_ignore_folder/factor_data_output/exp_new.pkl'
    with open(file_path, 'rb') as file:
        exp = pickle.load(file)
    exp = qlib_factor_runner.generate(exp)
    feedback = qlib_factor_summarizer.generateFeedback(exp, hypothesis, trace)
    # trace.hist.append((hypothesis, exp, feedback))
"""