"""
TODO: Model Structure RD-Loop
TODO: move the following code to a new class: Model_RD_Agent
"""

# import_from
from rdagent.app.model_proposal.conf import MODEL_PROP_SETTING
from rdagent.core.implementation import TaskGenerator
from rdagent.core.proposal import Belief2Task, BeliefSet, Imp2Feedback, Trace

# load_from_cls_uri


scen = load_from_cls_uri(MODEL_PROP_SETTING.scen)()

belief_gen = load_from_cls_uri(MODEL_PROP_SETTING.belief_gen)(scen)

belief2task: Belief2Task = load_from_cls_uri(MODEL_PROP_SETTING.belief2task)()

task_gen: TaskGenerator = load_from_cls_uri(MODEL_PROP_SETTING.task_gen)(scen)  # for implementation

imp2feedback: Imp2Feedback = load_from_cls_uri(MODEL_PROP_SETTING.imp2feedback)(scen)  # for implementation


iter_n = MODEL_PROP_SETTING.iter_n

trace = Trace()

belief_set = BeliefSet()
for _ in range(iter_n):
    belief = belief_gen.gen(trace)
    task = belief2task.convert(belief)
    imp = task_gen.gen(task)
    imp.execute()
    feedback = imp2feedback.summarize(imp)
    trace.hist.append((belief, feedback))
