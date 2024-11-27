import subprocess
from typing import Any, Literal

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import NextLoopException, RDLoop
from rdagent.core.exception import FactorEmptyError, ModelEmptyError
from rdagent.core.proposal import (
    Experiment2Feedback,
    ExpGen,
    Hypothesis2Experiment,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.log.time import measure_time
from rdagent.scenarios.kaggle.experiment.utils import python_files_to_notebook
from rdagent.scenarios.kaggle.kaggle_crawler import download_data


class DataScienceRDLoop(RDLoop):
    skip_loop_error = (NextLoopException,)

    @measure_time
    def __init__(self, PROP_SETTING: BasePropSetting):

        with logger.tag("init"):
            scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)
            logger.log_object(scen, tag="scenario")

            ### shared components in the workflow  # TODO: check if
            knowledge_base = (
                import_class(PROP_SETTING.knowledge_base)(PROP_SETTING.knowledge_base_path, scen)
                if PROP_SETTING.knowledge_base != ""
                else None
            )
            logger.log_object(knowledge_base, tag="knowledge_base")

            # 1) task generation from scratch
            # self.scratch_gen: tuple[HypothesisGen, Hypothesis2Experiment] = DummyHypothesisGen(scen),

            # 2) task generation from a complete solution
            self.exp_gen: ExpGen = import_class(PROP_SETTING.exp_gen)(scen)

            # self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
            # logger.log_object(self.hypothesis_gen, tag="hypothesis generator")
            # self.hypothesis2experiment: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
            # logger.log_object(self.hypothesis2experiment, tag="hypothesis2experiment")

            # TODO: we need more coder
            # self.feature_coder: Developer = import_class(PROP_SETTING.feature_coder)(scen)
            # logger.log_object(self.feature_coder, tag="feature coder")
            # self.model_feature_selection_coder: Developer = import_class(PROP_SETTING.model_feature_selection_coder)(
            #     scen
            # )
            # logger.log_object(self.model_feature_selection_coder, tag="model feature selection coder")
            # self.model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
            # logger.log_object(self.model_coder, tag="model coder")

            # TODO: now we only need on runner
            # self.feature_runner: Developer = import_class(PROP_SETTING.feature_runner)(scen)
            # logger.log_object(self.feature_runner, tag="feature runner")
            # self.model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)
            # logger.log_object(self.model_runner, tag="model runner")

            # self.summarizer: Experiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            # logger.log_object(self.summarizer, tag="summarizer")

            # self.trace = KGTrace(scen=scen, knowledge_base=knowledge_base)
            self.trace = Trace(scen=scen)
            super(RDLoop, self).__init__()

    @measure_time
    def direct_exp_gen(self, prev_out: dict[str, Any]):
        exp = self.exp_gen.gen(self.trace)
        hypo = exp.hypothesis
        return {"propose": hypo, "exp_gen": exp}

    @measure_time
    def coding(self, prev_out: dict[str, Any]):
        with logger.tag("d"):  # develop
            if prev_out["direct_exp_gen"]["propose"].action in [
                KG_ACTION_FEATURE_ENGINEERING,
                KG_ACTION_FEATURE_PROCESSING,
            ]:
                exp = self.feature_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
            elif prev_out["direct_exp_gen"]["propose"].action == KG_ACTION_MODEL_FEATURE_SELECTION:
                exp = self.model_feature_selection_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
            else:
                exp = self.model_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
            logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    @measure_time
    def running(self, prev_out: dict[str, Any]):
        if not self.exp_gen.is_complete():
            raise NextLoopExcpetion()

        with logger.tag("ef"):  # evaluate and feedback
            if prev_out["direct_exp_gen"]["propose"].action in [
                KG_ACTION_FEATURE_ENGINEERING,
                KG_ACTION_FEATURE_PROCESSING,
            ]:
                exp = self.feature_runner.develop(prev_out["coding"])
            else:
                exp = self.model_runner.develop(prev_out["coding"])
            logger.log_object(exp, tag="runner result")
            if DS_RD_SETTING.competition in [
                "optiver-realized-volatility-prediction",
                "covid19-global-forecasting-week-1",
            ]:
                try:
                    python_files_to_notebook(DS_RD_SETTING.competition, exp.experiment_workspace.workspace_path)
                except Exception as e:
                    logger.error(f"Merge python files to one file failed: {e}")
            if DS_RD_SETTING.auto_submit:
                csv_path = exp.experiment_workspace.workspace_path / "submission.csv"
                try:
                    subprocess.run(
                        [
                            "kaggle",
                            "competitions",
                            "submit",
                            "-f",
                            str(csv_path.absolute()),
                            "-m",
                            str(csv_path.parent.absolute()),
                            DS_RD_SETTING.competition,
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Auto submission failed: \n{e}")
                except Exception as e:
                    logger.error(f"Other exception when use kaggle api:\n{e}")

        return exp


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
        download_data(competition=DS_RD_SETTING.competition, local_path=DS_RD_SETTING.local_data_path)
    else:
        logger.error("Please specify competition name.")
    if path is None:
        kaggle_loop = DataScienceRDLoop(DS_RD_SETTING)
    else:
        kaggle_loop = DataScienceRDLoop.load(path)
    kaggle_loop.run(step_n=step_n)


if __name__ == "__main__":
    fire.Fire(main)
