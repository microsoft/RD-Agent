"""
TODO: Model Structure RD-Loop
TODO: move the following code to a new class: Model_RD_Agent
"""

# import_from
from rdagent.app.model_proposal.conf import PROP_SETTING
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis2Experiment,
    HypothesisGen,
    HypothesisSet,
    Trace,
)
from rdagent.core.task_generator import TaskGenerator
from rdagent.core.utils import import_class

# load_from_cls_uri

scen = import_class(PROP_SETTING.scen)()

hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)

hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()



# task_gen: TaskGenerator = load_from_cls_uri(PROP_SETTING.task_gen)(scen)  # for implementation

# imp2feedback: Experiment2Feedback = load_from_cls_uri(PROP_SETTING.imp2feedback)(scen)  # for implementation


trace = Trace(scen=scen)
hs = HypothesisSet()

hypothesis_set = HypothesisSet()
for _ in range(PROP_SETTING.evolving_n):
    hypothesis = hypothesis_gen.gen(trace)
    hypothesis2experiment.convert(hs)
