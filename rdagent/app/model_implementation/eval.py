from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent

from rdagent.components.task_implementation.model_implementation.benchmark.eval import ModelImpValEval
from rdagent.components.task_implementation.model_implementation.one_shot import ModelTaskGen
from rdagent.components.task_implementation.model_implementation.task import  ModelImpLoader, ModelTaskLoderJson


bench_folder = DIRNAME.parent.parent / "components" / "task_implementation" / "model_implementation" / "benchmark"
mtl = ModelTaskLoderJson(str(bench_folder / "model_dict.json"))

task_l = mtl.load()

task_l = [t for t in task_l if t.key == "A-DGN"]  # FIXME: other models does not work well

mtg = ModelTaskGen()

impl_l = mtg.generate(task_l)

# TODO: Align it with the benchmark framework after @wenjun's refine the evaluation part.
# Currently, we just handcraft a workflow for fast evaluation.

mil = ModelImpLoader(bench_folder / "gt_code")

mie = ModelImpValEval()
# Evaluation:
eval_l = []
for impl in impl_l:
    print(impl.target_task)
    gt_impl = mil.load(impl.target_task)
    eval_l.append(mie.evaluate(gt_impl, impl))

print(eval_l)
