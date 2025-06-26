import asyncio
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

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
from rdagent.core.exception import CoderError, PolicyError, RunnerError
from rdagent.core.proposal import ExperimentFeedback, ExpGen
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.dev.feedback import DSExperiment2Feedback
from rdagent.scenarios.data_science.dev.runner import DSCoSTEERRunner
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen import DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.idea_pool import DSKnowledgeBase
from rdagent.utils.workflow.misc import wait_retry


def clean_workspace(workspace_root: Path) -> None:
    """
    Clean the workspace folder and only keep the essential files to save more space.

    # remove all files and folders in the workspace except for .py, .md, and .csv files to avoid large workspace dump
    """
    for file_and_folder in workspace_root.iterdir():
        if file_and_folder.is_dir():
            if file_and_folder.is_symlink():
                file_and_folder.unlink()
            else:
                shutil.rmtree(file_and_folder)
        elif file_and_folder.is_file() and file_and_folder.suffix not in [".py", ".md", ".csv"]:
            file_and_folder.unlink()


@wait_retry()
def backup_folder(path: str | Path) -> Path:
    path = Path(path)
    workspace_bak_path = path.with_name(path.name + ".bak")
    if workspace_bak_path.exists():
        shutil.rmtree(workspace_bak_path)

    try:
        # `cp` may raise error if the workspace is beiing modified.
        # rsync is more robust choice, but it is not installed in some docker images.
        # use shutil.copytree(..., symlinks=True) should be more elegant, but it has more changes to raise error.
        subprocess.run(
            ["cp", "-r", "-P", str(path), str(workspace_bak_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error copying {path} to {workspace_bak_path}: {e}")
        logger.error(f"Stdout: {e.stdout.decode() if e.stdout else ''}")
        logger.error(f"Stderr: {e.stderr.decode() if e.stderr else ''}")
        raise
    return workspace_bak_path


class DataScienceRDLoop(RDLoop):
    # NOTE: we move the DataScienceRDLoop here to be easier to be imported
    skip_loop_error = (CoderError, RunnerError)
    withdraw_loop_error = (PolicyError,)

    @staticmethod
    def _get_exp_gen(class_uri: str, scen: Scenario):
        """
        Just for compatibility with the old version of the code.
        """
        # TODO: remove me in the future. I don't have to be this complicated.
        # It is just for compatibility with the old version of the code and configuration.
        from rdagent.scenarios.data_science.proposal.exp_gen.proposal import (
            DSProposalV1ExpGen,
            DSProposalV2ExpGen,
        )

        if class_uri == "rdagent.scenarios.data_science.proposal.exp_gen.DSExpGen":
            if DS_RD_SETTING.proposal_version not in ["v1", "v2"]:
                return import_class(DS_RD_SETTING.proposal_version)(scen=scen)
            if DS_RD_SETTING.proposal_version == "v1":
                return DSProposalV1ExpGen(scen=scen)
            if DS_RD_SETTING.proposal_version == "v2":
                return DSProposalV2ExpGen(scen=scen)
        return import_class(class_uri)(scen)

    def __init__(self, PROP_SETTING: BasePropSetting):
        logger.log_object(PROP_SETTING.competition, tag="competition")
        scen: Scenario = import_class(PROP_SETTING.scen)(PROP_SETTING.competition)

        # 1) task generation from scratch
        # self.scratch_gen: tuple[HypothesisGen, Hypothesis2Experiment] = DummyHypothesisGen(scen),

        # 2) task generation from a complete solution
        # self.exp_gen: ExpGen = import_class(PROP_SETTING.exp_gen)(scen)

        self.ckp_selector = import_class(PROP_SETTING.selector_name)()
        self.sota_exp_selector = import_class(PROP_SETTING.sota_exp_selector_name)()

        self.exp_gen: ExpGen = self._get_exp_gen(PROP_SETTING.hypothesis_gen, scen)

        # coders
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

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        # set the SOTA experiment to submit
        sota_exp_to_submit = self.sota_exp_selector.get_sota_exp_to_submit(self.trace)
        self.trace.set_sota_exp_to_submit(sota_exp_to_submit)

        # set the checkpoint to start from
        selection = self.ckp_selector.get_selection(self.trace)
        # set the current selection for the trace
        self.trace.set_current_selection(selection)

        # in parallel + multi-trace mode, the above global "trace.current_selection" will not be used
        # instead, we will use the "local_selection" attached to each exp to in async_gen().
        exp = await self.exp_gen.async_gen(self.trace, self)

        logger.log_object(exp)
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

        # set the local selection to the trace after feedback
        if exp.local_selection is not None:
            self.trace.set_current_selection(exp.local_selection)

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

        exp: DSExperiment = None

        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is None:
            exp = prev_out["running"]
            self.trace.hist.append((exp, prev_out["feedback"]))

            # NOTE: we put below  operations on selections here, instead of out of the if-else block,
            # to fit the corner case that the trace will be reset

            # set the local selection to the trace as global selection, then set the DAG parent for the trace
            if exp.local_selection is not None:
                self.trace.set_current_selection(exp.local_selection)
            self.trace.sync_dag_parent_and_hist()

        else:
            exp: DSExperiment = prev_out["direct_exp_gen"] if isinstance(e, CoderError) else prev_out["coding"]
            self.trace.hist.append(
                (
                    exp,
                    ExperimentFeedback.from_exception(e),
                )
            )

            # set the local selection to the trace as global selection, then set the DAG parent for the trace
            if exp.local_selection is not None:
                self.trace.set_current_selection(exp.local_selection)
            self.trace.sync_dag_parent_and_hist()

            if self.trace.sota_experiment() is None:
                if DS_RD_SETTING.coder_on_whole_pipeline:
                    #  check if feedback is not generated
                    if len(self.trace.hist) >= DS_RD_SETTING.coding_fail_reanalyze_threshold:
                        recent_hist = self.trace.hist[-DS_RD_SETTING.coding_fail_reanalyze_threshold :]
                        if all(isinstance(fb.exception, (CoderError, RunnerError)) for _, fb in recent_hist):
                            new_scen = self.trace.scen
                            if hasattr(new_scen, "reanalyze_competition_description"):
                                logger.info(
                                    "Reanalyzing the competition description after three consecutive coding failures."
                                )
                                new_scen.reanalyze_competition_description()
                                self.trace.scen = new_scen
                            else:
                                logger.info("Can not reanalyze the competition description.")
                elif len(self.trace.hist) >= DS_RD_SETTING.consecutive_errors:
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
            log_back_path = backup_folder(Path().cwd() / "log")
            subprocess.run(["tar", "-cf", str(mid_log_tar_path), "-C", str(log_back_path), "."], check=True)

            # only clean current workspace without affecting other loops.
            for k in "direct_exp_gen", "coding", "running":
                if k in prev_out:
                    assert isinstance(prev_out[k], DSExperiment)
                    clean_workspace(prev_out[k].experiment_workspace.workspace_path)

            # Backup the workspace (only necessary files are included)
            # - Step 1: Copy the workspace to a .bak package
            workspace_bak_path = backup_folder(RD_AGENT_SETTINGS.workspace_path)

            # - Step 2: Clean .bak package
            for bak_workspace in workspace_bak_path.iterdir():
                clean_workspace(bak_workspace)

            # - Step 3: Create tarball from the cleaned .bak workspace
            subprocess.run(["tar", "-cf", str(mid_workspace_tar_path), "-C", str(workspace_bak_path), "."], check=True)

            # - Step 4: Remove .bak package
            shutil.rmtree(workspace_bak_path)

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
        path: str | Path,
        checkout: bool | str | Path = False,
        replace_timer: bool = True,
    ) -> "LoopBase":
        session = super().load(path, checkout, replace_timer)
        logger.log_object(DS_RD_SETTING.competition, tag="competition")  # NOTE: necessary to make mle_summary work.
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
