from pathlib import Path
from typing import Any

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.ensemble import EnsembleCoSTEER
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature import FeatureCoSTEER
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model import ModelCoSTEER
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.raw_data_loader import DataLoaderCoSTEER
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.workflow import WorkflowCoSTEER
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import CoderError, RunnerError
from rdagent.core.proposal import ExperimentFeedback
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.dev.feedback import DSExperiment2Feedback
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSExpGen, DSTrace
from rdagent.scenarios.kaggle.kaggle_crawler import download_data


class DataScienceRDLoop(RDLoop):
    skip_loop_error = (CoderError, RunnerError)

    def __init__(self, PROP_SETTING: BasePropSetting):
        logger.log_object(PROP_SETTING.competition, tag="competition")
        scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)

        ### shared components in the workflow  # TODO: check if
        knowledge_base = (
            import_class(PROP_SETTING.knowledge_base)(PROP_SETTING.knowledge_base_path, scen)
            if PROP_SETTING.knowledge_base != ""
            else None
        )

        # 1) task generation from scratch
        # self.scratch_gen: tuple[HypothesisGen, Hypothesis2Experiment] = DummyHypothesisGen(scen),

        # 2) task generation from a complete solution
        # self.exp_gen: ExpGen = import_class(PROP_SETTING.exp_gen)(scen)
        self.exp_gen = DSExpGen(scen)
        self.data_loader_coder = DataLoaderCoSTEER(scen)
        self.feature_coder = FeatureCoSTEER(scen)
        self.model_coder = ModelCoSTEER(scen)
        self.ensemble_coder = EnsembleCoSTEER(scen)
        self.workflow_coder = WorkflowCoSTEER(scen)

        self.runner = DSCoSTEERRunner(scen)
        # self.summarizer: Experiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
        # logger.log_object(self.summarizer, tag="summarizer")

        # self.trace = KGTrace(scen=scen, knowledge_base=knowledge_base)
        self.trace = DSTrace(scen=scen)
        self.summarizer = DSExperiment2Feedback(scen)
        super(RDLoop, self).__init__()

    def direct_exp_gen(self, prev_out: dict[str, Any]):
        exp = self.exp_gen.gen(self.trace)
        logger.log_object(exp)

        # FIXME: this is for LLM debug webapp, remove this when the debugging is done.
        logger.log_object(exp, tag="debug_exp_gen")
        return exp

    def coding(self, prev_out: dict[str, Any]):
        exp = prev_out["direct_exp_gen"]
        for tasks in exp.pending_tasks_list:
            exp.sub_tasks = tasks
            if isinstance(exp.sub_tasks[0], DataLoaderTask):
                exp = self.data_loader_coder.develop(exp)
            elif isinstance(exp.sub_tasks[0], FeatureTask):
                exp = self.feature_coder.develop(exp)
            elif isinstance(exp.sub_tasks[0], ModelTask):
                exp = self.model_coder.develop(exp)
            elif isinstance(exp.sub_tasks[0], EnsembleTask):
                exp = self.ensemble_coder.develop(exp)
            elif isinstance(exp.sub_tasks[0], WorkflowTask):
                exp = self.workflow_coder.develop(exp)
            else:
                raise NotImplementedError(f"Unsupported component in DataScienceRDLoop: {exp.hypothesis.component}")
            exp.sub_tasks = []
        logger.log_object(exp)
        return exp

    def running(self, prev_out: dict[str, Any]):
        exp: DSExperiment = prev_out["coding"]
        if exp.next_component_required() is None:
            new_exp = self.runner.develop(exp)
            logger.log_object(new_exp)
            return new_exp
        else:
            return exp

    def feedback(self, prev_out: dict[str, Any]) -> ExperimentFeedback:
        exp: DSExperiment = prev_out["running"]
        if exp.next_component_required() is None:
            feedback = self.summarizer.generate_feedback(exp, self.trace)
        else:
            feedback = ExperimentFeedback(
                reason=f"{exp.hypothesis.component} is completed.",
                decision=True,
            )
        logger.log_object(feedback)
        return feedback

    def record(self, prev_out: dict[str, Any]):
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is None:
            self.trace.hist.append((prev_out["running"], prev_out["feedback"]))
        else:
            self.trace.hist.append(
                (
                    prev_out["direct_exp_gen"] if isinstance(e, CoderError) else prev_out["coding"],
                    ExperimentFeedback.from_exception(e),
                )
            )
            if self.trace.sota_experiment() is None and len(self.trace.hist) >= DS_RD_SETTING.consecutive_errors:
                trace_exp_next_component_list = [
                    type(exp.pending_tasks_list[0][0])
                    for exp, _ in self.trace.hist[-DS_RD_SETTING.consecutive_errors :]
                ]
                last_successful_exp = self.trace.last_successful_exp()
                if (
                    last_successful_exp not in [exp for exp, _ in self.trace.hist[-DS_RD_SETTING.consecutive_errors :]]
                    and len(set(trace_exp_next_component_list)) == 1
                ):
                    logger.error("Consecutive errors reached the limit. Dumping trace.")
                    logger.log_object(self.trace, tag="trace before restart")
                    self.trace = DSTrace(scen=self.trace.scen, knowledge_base=self.trace.knowledge_base)
        logger.log_object(self.trace, tag="trace")
        logger.log_object(self.trace.sota_experiment(), tag="SOTA experiment")


def main(path=None, step_n=None, loop_n=None, competition="bms-molecular-translation"):
    """

    Parameters
    ----------
    path :
        path like `$LOG_PATH/__session__/1/0_propose`. It indicates that we restore the state that after finish the step 0 in loop1
    step_n :
        How many steps to run; if None, it will run forever until error or KeyboardInterrupt
    loop_n :
        How many loops to run; if None, it will run forever until error or KeyboardInterrupt
        - if current loop is incomplete, it will be counted as the first loop for completion.
        - if both step_n and loop_n are provided, the process will stop as soon as either condition is met.
    competition :


    Auto R&D Evolving loop for models in a Kaggle scenario.
    You can continue running session by
    .. code-block:: bash
        dotenv run -- python rdagent/app/data_science/loop.py [--competition titanic] $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional parameter
        rdagent kaggle --competition playground-series-s4e8  # You are encouraged to use this one.
    """
    if competition is not None:
        DS_RD_SETTING.competition = competition

    if DS_RD_SETTING.competition:
        if DS_RD_SETTING.scen.endswith("KaggleScen"):
            download_data(competition=DS_RD_SETTING.competition, settings=DS_RD_SETTING)
        else:
            if not Path(f"{DS_RD_SETTING.local_data_path}/{competition}").exists():
                logger.error(f"Please prepare data for competition {competition} first.")
                return
    else:
        logger.error("Please specify competition name.")
    if path is None:
        kaggle_loop = DataScienceRDLoop(DS_RD_SETTING)
    else:
        kaggle_loop = DataScienceRDLoop.load(path)
    kaggle_loop.run(step_n=step_n, loop_n=loop_n)


if __name__ == "__main__":
    fire.Fire(main)
