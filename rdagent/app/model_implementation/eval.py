from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent

from rdagent.model_implementation.benchmark.eval import ModelImpValEval
from rdagent.model_implementation.one_shot import ModelTaskGen
from rdagent.model_implementation.task import ModelImpLoader, ModelTaskLoderJson

mtl = ModelTaskLoderJson("TODO: A Path to json")

task_l = mtl.load()

mtg = ModelTaskGen()

impl_l = mtg.generate(task_l)

# TODO: Align it with the benchmark framework after @wenjun's refine the evaluation part.
# Currently, we just handcraft a workflow for fast evaluation.

mil = ModelImpLoader(DIRNAME.parent.parent / "model_implementation" / "benchmark" /  "gt_code")

mie = ModelImpValEval()
# Evaluation:
eval_l = []
for impl in impl_l:
    gt_impl = mil.load(impl.target_task)
    eval_l.append(mie.evaluate(gt_impl, impl))

print(eval_l)
