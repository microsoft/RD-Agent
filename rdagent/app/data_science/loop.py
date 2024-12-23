import subprocess
from typing import Any, Literal

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
from rdagent.core.exception import FactorEmptyError, ModelEmptyError
from rdagent.core.proposal import (
    Experiment2Feedback,
    ExpGen,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.dev.feedback import DSExperiment2Feedback
from rdagent.scenarios.data_science.dev.runner import DSRunner
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSExpGen, DSTrace
from rdagent.scenarios.kaggle.kaggle_crawler import download_data


class DataScienceRDLoop(RDLoop):

    def __init__(self, PROP_SETTING: BasePropSetting):
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

        self.runner = DSRunner(scen)
        # self.summarizer: Experiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
        # logger.log_object(self.summarizer, tag="summarizer")

        # self.trace = KGTrace(scen=scen, knowledge_base=knowledge_base)
        self.trace = DSTrace(scen=scen)
        self.summarizer = DSExperiment2Feedback(scen)
        super(RDLoop, self).__init__()

    def direct_exp_gen(self, prev_out: dict[str, Any]):
        exp = self.exp_gen.gen(self.trace)
        return exp

    def coding(self, prev_out: dict[str, Any]):
        exp: DSExperiment = prev_out["direct_exp_gen"]
        exp_task = exp.sub_tasks[0]
        if isinstance(exp_task, DataLoaderTask):
            exp = self.data_loader_coder.develop(exp)
        elif isinstance(exp_task, FeatureTask):
            exp = self.feature_coder.develop(exp)
        elif isinstance(exp_task, ModelTask):
            exp = self.model_coder.develop(exp)
        elif isinstance(exp_task, EnsembleTask):
            exp = self.ensemble_coder.develop(exp)
        elif isinstance(exp_task, WorkflowTask):
            exp = self.workflow_coder.develop(exp)
        else:
            raise NotImplementedError(f"Unsupported task type in DataScienceRDLoop: {exp_task}")

        return exp

    def running(self, prev_out: dict[str, Any]):
        if self.trace.all_components_completed():
            exp = self.runner.develop(prev_out["coding"])
        else:
            exp = prev_out["coding"]
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        if self.trace.all_components_completed():
            feedback = self.summarizer.generate_feedback(
                prev_out["running"], prev_out["direct_exp_gen"].hypothesis, self.trace
            )
        else:
            feedback = HypothesisFeedback(
                observations="Not all 5 components are completed, skip feedback of DataScienceRDLoop.",
                hypothesis_evaluation="",
                new_hypothesis="",
                reason="",
                decision=True,
            )
        self.trace.hist.append((prev_out["direct_exp_gen"].hypothesis, prev_out["running"], feedback))


def main(path=None, step_n=None, competition=None):
    """
    Auto R&D Evolving loop for models in a kaggle{} scenario.
    You can continue running session by
    .. code-block:: bash
        dotenv run -- python rdagent/app/data_science/loop.py [--competition titanic] $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` is a optional parameter
        rdagent kaggle --competition playground-series-s4e8  # You are encouraged to use this one.
    """
    if competition is not None:
        DS_RD_SETTING.competition = competition

    if DS_RD_SETTING.competition:
        download_data(competition=DS_RD_SETTING.competition, settings=DS_RD_SETTING)
    else:
        logger.error("Please specify competition name.")
    if path is None:
        kaggle_loop = DataScienceRDLoop(DS_RD_SETTING)
    else:
        kaggle_loop = DataScienceRDLoop.load(path)
    kaggle_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
