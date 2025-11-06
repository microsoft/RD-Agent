import asyncio
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from rdagent.app.agentic_sys.conf import AgenticSysSetting
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import CoderError, PolicyError, RunnerError
from rdagent.core.proposal import ExperimentFeedback, ExpGen
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger

from rdagent.core.proposal import ExpGen


class AgenticSysRDLoop(RDLoop):
    # NOTE: we move the DataScienceRDLoop here to be easier to be imported
    skip_loop_error = (CoderError, RunnerError)
    withdraw_loop_error = (PolicyError,)

    def __init__(self, PROP_SETTING: AgenticSysSetting):

        scen = import_class(PROP_SETTING.scen)()
        self.scen: Scenario = scen
        self.exp_gen: ExpGen = import_class(PROP_SETTING.hypothesis_gen)(scen)

        self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
        self.runner: Developer = import_class(PROP_SETTING.runner)(scen)

        self.feedback: Experiment2Feedback = import_class(PROP_SETTING.feedback)(scen)
        self.trace = Trace(scen=scen)

        super(RDLoop, self).__init__()

    async def _exp_gen(self, prev_out: dict[str, Any]):


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

        cur_loop_id = prev_out[self.LOOP_IDX_KEY]

        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is None:
            exp = prev_out["running"]

            # NOTE: we put below  operations on selections here, instead of out of the if-else block,
            # to fit the corner case that the trace will be reset

            # set the local selection to the trace as global selection, then set the DAG parent for the trace
            if exp.local_selection is not None:
                self.trace.set_current_selection(exp.local_selection)
            self.trace.sync_dag_parent_and_hist((exp, prev_out["feedback"]), cur_loop_id)
        else:
            exp: DSExperiment = prev_out["direct_exp_gen"] if isinstance(e, CoderError) else prev_out["coding"]
            # TODO: distinguish timeout error & other exception.
            if (
                isinstance(self.trace.scen, DataScienceScen)
                and DS_RD_SETTING.allow_longer_timeout
                and isinstance(e, CoderError)
                and e.caused_by_timeout
            ):
                logger.info(
                    f"Timeout error occurred: {e}. Increasing timeout for the current scenario from {self.trace.scen.timeout_increase_count} to {self.trace.scen.timeout_increase_count + 1}."
                )
                self.trace.scen.increase_timeout()

            # set the local selection to the trace as global selection, then set the DAG parent for the trace
            if exp.local_selection is not None:
                self.trace.set_current_selection(exp.local_selection)

            self.trace.sync_dag_parent_and_hist(
                (
                    exp,
                    ExperimentFeedback.from_exception(e),
                ),
                cur_loop_id,
            )
            # Value backpropagation is handled in async_gen before next() via observe_commits

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
                        # Reset the trace; MCTS stats will be cleared via registered callback
                        self.exp_gen.reset()

        # set the SOTA experiment to submit
        sota_exp_to_submit = self.sota_exp_selector.get_sota_exp_to_submit(self.trace)
        self.trace.set_sota_exp_to_submit(sota_exp_to_submit)
        logger.log_object(sota_exp_to_submit, tag="sota_exp_to_submit")

        logger.log_object(self.trace, tag="trace")
        logger.log_object(self.trace.sota_experiment(search_type="all"), tag="SOTA experiment")

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
                if k in prev_out and prev_out[k] is not None:
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

    def _check_exit_conditions_on_step(self, loop_id: Optional[int] = None, step_id: Optional[int] = None):
        if step_id not in [self.steps.index("running"), self.steps.index("feedback")]:
            # pass the check for running and feedbacks since they are very likely to be finished soon.
            super()._check_exit_conditions_on_step(loop_id=loop_id, step_id=step_id)

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
