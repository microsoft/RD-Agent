import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import fire

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.ensemble import EnsembleCoSTEER
from rdagent.components.coder.data_science.ensemble.exp import EnsembleTask
from rdagent.components.coder.data_science.feature import FeatureCoSTEER
from rdagent.components.coder.data_science.feature.exp import FeatureTask
from rdagent.components.coder.data_science.model import ModelCoSTEER
from rdagent.components.coder.data_science.model.exp import ModelTask
from rdagent.components.coder.data_science.pipeline import PipelineCoSTEER
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.raw_data_loader import DataLoaderCoSTEER
from rdagent.components.coder.data_science.raw_data_loader.exp import DataLoaderTask
from rdagent.components.coder.data_science.share.doc import DocDev
from rdagent.components.coder.data_science.workflow import WorkflowCoSTEER
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import CoderError, RunnerError
from rdagent.core.proposal import ExperimentFeedback
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.dev.feedback import DSExperiment2Feedback
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSExpGen, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.idea_pool import DSKnowledgeBase
from rdagent.scenarios.data_science.proposal.exp_gen.select import LatestCKPSelector
from rdagent.scenarios.kaggle.kaggle_crawler import download_data


class DataScienceRDLoop(RDLoop):
    skip_loop_error = (CoderError, RunnerError)

    def __init__(self, PROP_SETTING: BasePropSetting):
        logger.log_object(PROP_SETTING.competition, tag="competition")
        scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)

        # 1) task generation from scratch
        # self.scratch_gen: tuple[HypothesisGen, Hypothesis2Experiment] = DummyHypothesisGen(scen),

        # 2) task generation from a complete solution
        # self.exp_gen: ExpGen = import_class(PROP_SETTING.exp_gen)(scen)
        self.ckp_selector = LatestCKPSelector()
        self.exp_gen = DSExpGen(scen)
        self.data_loader_coder = DataLoaderCoSTEER(scen)
        self.feature_coder = FeatureCoSTEER(scen)
        self.model_coder = ModelCoSTEER(scen)
        self.ensemble_coder = EnsembleCoSTEER(scen)
        self.workflow_coder = WorkflowCoSTEER(scen)

        self.pipeline_coder = PipelineCoSTEER(scen)

        self.runner = DSCoSTEERRunner(scen)
        if DS_RD_SETTING.enable_doc_dev:
            self.docdev = DocDev(scen)
        # self.summarizer: Experiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
        # logger.log_object(self.summarizer, tag="summarizer")

        if DS_RD_SETTING.enable_knowledge_base and DS_RD_SETTING.knowledge_base_version == "v1":
            knowledge_base = DSKnowledgeBase(
                path=DS_RD_SETTING.knowledge_base_path, idea_pool_json_path=DS_RD_SETTING.idea_pool_json_path
            )
            self.trace = DSTrace(scen=scen, knowledge_base=knowledge_base)
        else:
            self.trace = DSTrace(scen=scen)
        self.summarizer = DSExperiment2Feedback(scen)
        super(RDLoop, self).__init__()

    def direct_exp_gen(self, prev_out: dict[str, Any]):
        selection = self.ckp_selector.get_selection(self.trace)
        exp = self.exp_gen.gen(self.trace, selection)
        logger.log_object(exp)

        # FIXME: this is for LLM debug webapp, remove this when the debugging is done.
        logger.log_object(exp, tag="debug_exp_gen")
        return exp

    def coding(self, prev_out: dict[str, Any]):
        exp = prev_out["direct_exp_gen"]
        for tasks in exp.pending_tasks_list:
            exp.sub_tasks = tasks
            with logger.tag(f"{exp.sub_tasks[0].__class__.__name__}"):
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
                elif isinstance(exp.sub_tasks[0], PipelineTask):
                    exp = self.pipeline_coder.develop(exp)
                else:
                    raise NotImplementedError(f"Unsupported component in DataScienceRDLoop: {exp.hypothesis.component}")
            exp.sub_tasks = []
        logger.log_object(exp)
        return exp

    def running(self, prev_out: dict[str, Any]):
        exp: DSExperiment = prev_out["coding"]
        if exp.is_ready_to_run():
            new_exp = self.runner.develop(exp)
            logger.log_object(new_exp)
            exp = new_exp
        if DS_RD_SETTING.enable_doc_dev:
            self.docdev.develop(exp)
        return exp

    def feedback(self, prev_out: dict[str, Any]) -> ExperimentFeedback:
        """
        Assumption:
        - If we come to feedback phase, the previous development steps are successful.
        """
        exp: DSExperiment = prev_out["running"]
        if self.trace.next_incomplete_component() is None or DS_RD_SETTING.coder_on_whole_pipeline:
            # we have alreadly completed components in previous trace. So current loop is focusing on a new proposed idea.
            # So we need feedback for the proposal.
            feedback = self.summarizer.generate_feedback(exp, self.trace)
        else:
            # Otherwise, it is on drafting stage, don't need complicated feedbacks.
            feedback = ExperimentFeedback(
                reason=f"{exp.hypothesis.component} is completed.",
                decision=True,
            )
        logger.log_object(feedback)
        return feedback

    def record(self, prev_out: dict[str, Any]):

        # set the DAG parent for the trace
        self.trace.sync_dag_parent_and_hist()

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
            if (
                self.trace.sota_experiment() is None
                and len(self.trace.hist) >= DS_RD_SETTING.consecutive_errors
                and not DS_RD_SETTING.coder_on_whole_pipeline
            ):
                # if {in inital/drafting stage} and {tried enough times}
                for _, fb in self.trace.hist[-DS_RD_SETTING.consecutive_errors :]:
                    if fb:
                        break  # any success will stop restarting.
                else:  # otherwise restart it
                    logger.error("Consecutive errors reached the limit. Dumping trace.")
                    logger.log_object(self.trace, tag="trace before restart")
                    self.trace = DSTrace(scen=self.trace.scen, knowledge_base=self.trace.knowledge_base)
        logger.log_object(self.trace, tag="trace")
        logger.log_object(self.trace.sota_experiment(), tag="SOTA experiment")
        if DS_RD_SETTING.enable_knowledge_base and DS_RD_SETTING.knowledge_base_version == "v1":
            logger.log_object(self.trace.knowledge_base, tag="knowledge_base")
            self.trace.knowledge_base.dump()

        if (
            DS_RD_SETTING.enable_log_archive
            and DS_RD_SETTING.log_archive_path is not None
            and Path(DS_RD_SETTING.log_archive_path).is_dir()
        ):
            start_archive_datetime = datetime.now()
            logger.info(f"Archiving log and workspace folder after loop {self.loop_idx}")
            mid_log_tar_path = (
                Path(
                    DS_RD_SETTING.log_archive_temp_path
                    if DS_RD_SETTING.log_archive_temp_path
                    else DS_RD_SETTING.log_archive_path
                )
                / "mid_log.tar"
            )
            mid_workspace_tar_path = (
                Path(
                    DS_RD_SETTING.log_archive_temp_path
                    if DS_RD_SETTING.log_archive_temp_path
                    else DS_RD_SETTING.log_archive_path
                )
                / "mid_workspace.tar"
            )
            subprocess.run(["tar", "-cf", str(mid_log_tar_path), "-C", (Path().cwd() / "log"), "."], check=True)

            # remove all files and folders in the workspace except for .py, .md, and .csv files to avoid large workspace dump
            for workspace_id in Path(RD_AGENT_SETTINGS.workspace_path).iterdir():
                for file_and_folder in workspace_id.iterdir():
                    if file_and_folder.is_dir():
                        shutil.rmtree(file_and_folder)
                    elif file_and_folder.is_file() and file_and_folder.suffix not in [".py", ".md", ".csv"]:
                        file_and_folder.unlink()

            subprocess.run(
                ["tar", "-cf", str(mid_workspace_tar_path), "-C", (RD_AGENT_SETTINGS.workspace_path), "."], check=True
            )
            if DS_RD_SETTING.log_archive_temp_path is not None:
                shutil.move(mid_log_tar_path, Path(DS_RD_SETTING.log_archive_path) / "mid_log.tar")
                mid_log_tar_path = Path(DS_RD_SETTING.log_archive_path) / "mid_log.tar"
                shutil.move(mid_workspace_tar_path, Path(DS_RD_SETTING.log_archive_path) / "mid_workspace.tar")
                mid_workspace_tar_path = Path(DS_RD_SETTING.log_archive_path) / "mid_workspace.tar"
            shutil.copy(
                mid_log_tar_path, Path(DS_RD_SETTING.log_archive_path) / "mid_log_bak.tar"
            )  # backup when upper code line is killed when running
            shutil.copy(
                mid_workspace_tar_path, Path(DS_RD_SETTING.log_archive_path) / "mid_workspace_bak.tar"
            )  # backup when upper code line is killed when running
            self.timer.add_duration(datetime.now() - start_archive_datetime)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        do_truncate: bool = False,
        replace_timer: bool = True,
    ) -> "LoopBase":
        session = super().load(path, output_path, do_truncate, replace_timer)
        if DS_RD_SETTING.enable_knowledge_base and DS_RD_SETTING.knowledge_base_version == "v1":
            session.trace.knowledge_base = DSKnowledgeBase(
                path=DS_RD_SETTING.knowledge_base_path, idea_pool_json_path=DS_RD_SETTING.idea_pool_json_path
            )
        return session

    def dump(self, path: str | Path) -> None:
        """
        Since knowledge_base is big and we don't want to dump it every time
        So we remove it from the trace before dumping and restore it after.
        """
        backup_knowledge_base = None
        if self.trace.knowledge_base is not None:
            backup_knowledge_base = self.trace.knowledge_base
            self.trace.knowledge_base = None
        super().dump(path)
        if backup_knowledge_base is not None:
            self.trace.knowledge_base = backup_knowledge_base


def main(
    path=None,
    output_path=None,
    step_n=None,
    loop_n=None,
    competition="bms-molecular-translation",
    do_truncate=True,
    timeout=None,
    replace_timer=True,
):
    """

    Parameters
    ----------
    path :
        path like `$LOG_PATH/__session__/1/0_propose`. It indicates that we restore the state that after finish the step 0 in loop 1
    output_path :
        path like `$LOG_PATH`. It indicates that where we want to save our session and log information.
    step_n :
        How many steps to run; if None, it will run forever until error or KeyboardInterrupt
    loop_n :
        How many loops to run; if None, it will run forever until error or KeyboardInterrupt
        - if current loop is incomplete, it will be counted as the first loop for completion.
        - if both step_n and loop_n are provided, the process will stop as soon as either condition is met.
    competition :
    do_truncate :
        If set to True, the logger will truncate the future log messages by calling `logger.storage.truncate`.


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
        kaggle_loop = DataScienceRDLoop.load(path, output_path, do_truncate, replace_timer)
    kaggle_loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout)


if __name__ == "__main__":
    fire.Fire(main)
