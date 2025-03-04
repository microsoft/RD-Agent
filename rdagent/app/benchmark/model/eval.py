from pathlib import Path
from rdagent.components.benchmark.conf import BenchmarkSettings
from rdagent.components.coder.model_coder import ModelCoSTEER
from rdagent.components.loader.task_loader import ModelTaskLoaderJson, ModelWsLoader
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)

if __name__ == "__main__":
    DIRNAME = Path(__file__).absolute().resolve().parent
    # 1.read the settings
    bs = BenchmarkSettings()

    from rdagent.components.coder.model_coder.benchmark.eval import ModelImpValEval
    from rdagent.components.coder.model_coder.one_shot import ModelCodeWriter

    bench_folder = DIRNAME.parent.parent.parent / "components" / "coder" / "model_coder" / "benchmark"
    model_task_loader = ModelTaskLoaderJson(str(bench_folder / "model_dict.json"))

    task_l = model_task_loader.load()

    task_l = [t for t in task_l if t.name == "A-DGN"]  # FIXME: other models does not work well
    # task_l = [t for t in task_l if t.name == "PMLP"]  # FIXME: other models does not work well

    model_experiment = QlibModelExperiment(sub_tasks=task_l)
    model_target_generator = ModelCodeWriter(scen=QlibModelScenario())
    # mtg = ModelCoSTEER(scen=QlibModelScenario())

    model_experiment = model_target_generator.develop(model_experiment)

    # TODO: Align it with the benchmark framework after @wenjun's refine the evaluation part.
    # Currently, we just handcraft a workflow for fast evaluation.

    gt_impl_loader = ModelWsLoader(bench_folder / "gt_code")

    model_imp_evaluator = ModelImpValEval()
    # Evaluation:
    eval_result_l = []
    for impl in model_experiment.sub_workspace_list:
        print(impl.target_task)
        gt_impl = gt_impl_loader.load(impl.target_task)
        print (gt_impl)
        try:
            eval_result_l.append(model_imp_evaluator.evaluate(gt_impl, impl))
        except Exception as e:
            print(f"Error in evaluating model: {e}")
            eval_result_l.append(None)
    print(eval_result_l)
