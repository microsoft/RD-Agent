from pathlib import Path

from rdagent.components.coder.model_coder import ModelCoSTEER
from rdagent.components.loader.task_loader import ModelTaskLoaderJson, ModelWsLoader
from torch_geometric.nn.conv import MessagePassing
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)

if __name__ == "__main__":
    DIRNAME = Path(__file__).absolute().resolve().parent

    from rdagent.components.coder.model_coder.benchmark.eval import ModelImpValEval
    from rdagent.components.coder.model_coder.one_shot import ModelCodeWriter

    bench_folder = DIRNAME.parent.parent.parent / "components" / "coder" / "model_coder" / "benchmark"
    mtl = ModelTaskLoaderJson(str(bench_folder / "model_dict.json"))

    task_l = mtl.load()

    # task_l = [t for t in task_l if t.name == "A-DGN"]  # FIXME: other models does not work well
    # task_l = [t for t in task_l if t.name == "PMLP"]  
    model_experiment = QlibModelExperiment(sub_tasks=task_l)
    mtg = ModelCodeWriter(scen=QlibModelScenario())
    # mtg = ModelCoSTEER(scen=QlibModelScenario())

    model_experiment = mtg.develop(model_experiment)

    # TODO: Align it with the benchmark framework after @wenjun's refine the evaluation part.
    # Currently, we just handcraft a workflow for fast evaluation.

    mil = ModelWsLoader(bench_folder / "gt_code")

    mie = ModelImpValEval()
    # Evaluation:
    eval_l = []
    eval_d = {}
    for impl in model_experiment.sub_workspace_list:
        print(impl.target_task.name)
        try:
            gt_impl = mil.load(impl.target_task)
            eval_l.append(mie.evaluate(gt_impl, impl, impl.target_task.name))
            eval_d[impl.target_task.name] = eval_l[-1]
        except Exception as e:
            print(f"Error in {impl.target_task.name}: {e}")
    print(eval_l)
    print(eval_d)
