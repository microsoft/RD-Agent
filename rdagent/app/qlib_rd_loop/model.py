"""
TODO: Model Structure RD-Loop
TODO: move the following code to a new class: Model_RD_Agent
"""

# import_from
from rdagent.app.model_proposal.conf import MODEL_PROP_SETTING

from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis2Experiment,
    HypothesisSet,
    Trace,
)
from rdagent.core.task_generator import TaskGenerator

# load_from_cls_uri


scen = load_from_cls_uri(MODEL_PROP_SETTING.scen)()

hypothesis_gen = load_from_cls_uri(MODEL_PROP_SETTING.hypothesis_gen)(scen)

hypothesis2task: Hypothesis2Experiment = load_from_cls_uri(MODEL_PROP_SETTING.hypothesis2task)()

task_gen: TaskGenerator = load_from_cls_uri(MODEL_PROP_SETTING.task_gen)(scen)  # for implementation

imp2feedback: Experiment2Feedback = load_from_cls_uri(MODEL_PROP_SETTING.imp2feedback)(scen)  # for implementation


iter_n = MODEL_PROP_SETTING.iter_n

trace = Trace()

hypothesis_set = HypothesisSet()
for _ in range(iter_n):
    hypothesis = hypothesis_gen.gen(trace)
    task = hypothesis2task.convert(hypothesis)
    imp = task_gen.gen(task)
    imp.execute()
    feedback = imp2feedback.summarize(imp)
    trace.hist.append((hypothesis, feedback))
