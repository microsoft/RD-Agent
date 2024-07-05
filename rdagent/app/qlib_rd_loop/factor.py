"""
TODO: Factor Structure RD-Loop
"""

from dotenv import load_dotenv

load_dotenv(override=True)

# import_from
from rdagent.app.qlib_rd_loop.conf import PROP_SETTING
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis2Experiment,
    HypothesisGen,
    HypothesisSet,
    Trace,
)
from rdagent.core.task_generator import TaskGenerator
from rdagent.core.utils import import_class

scen = import_class(PROP_SETTING.scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()

qlib_factor_coder: TaskGenerator = import_class(PROP_SETTING.qlib_factor_coder)(scen)
qlib_factor_runner: TaskGenerator = import_class(PROP_SETTING.qlib_factor_runner)(scen)

qlib_factor_summarizer: Experiment2Feedback = import_class(PROP_SETTING.qlib_factor_summarizer)()


trace = Trace(scen=scen)
hs = HypothesisSet(trace=trace)
for _ in range(PROP_SETTING.evolving_n):
    hypothesis = hypothesis_gen.gen(trace)
    exp = hypothesis2experiment.convert(hs)
    exp = qlib_factor_coder.generate(exp)
    exp = qlib_factor_runner.generate(exp)
    feedback = qlib_factor_summarizer.summarize(exp)

    trace.hist.append((hypothesis, exp, feedback))
