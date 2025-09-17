import json
import pickle
import time
import uuid
from abc import abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.experiment import Task
from rdagent.core.interactor import Interactor
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.utils.agent.tpl import T


class DSInteractor(Interactor[DSExperiment]):
    @abstractmethod
    def dump_and_wait_for_user_input(
        self,
        scenario_description: str,
        ds_trace_desc: str,
        current_code: str,
        hypothesis_candidates: list[str],
        target_hypothesis: DSHypothesis,
        target_hypothesis_index: int,
        task_description: Task,
        exp: DSExperiment,
    ) -> DSExperiment:
        raise NotImplementedError

    def interact(self, exp: DSExperiment, trace: DSTrace) -> DSExperiment:
        """
        Interact with the experiment to get feedback or confirmation.

        Responsibilities:
        - Present the current state of the experiment.
        - Collect input to guide the next steps in the experiment.
        - Rewrite the experiment based on feedback.
        """
        scenario_description = self.scen.get_scenario_all_desc(
            eda_output=exp.experiment_workspace.file_dict.get("EDA.md", None)
        )
        ds_trace_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=trace.experiment_and_feedback_list_after_init(return_type="all"),
            type="all",
            pipeline=DS_RD_SETTING.coder_on_whole_pipeline,
        )
        current_code = exp.experiment_workspace.file_dict.get("main.py", "")
        target_hypothesis = exp.hypothesis

        hypothesis_str_candidates = [hypo.hypothesis for hypo in exp.hypothesis_candidates]
        target_hypothesis_index = (
            hypothesis_str_candidates.index(target_hypothesis.hypothesis)
            if target_hypothesis.hypothesis in hypothesis_str_candidates and not trace.is_selection_new_tree()
            else -1
        )
        return self.dump_and_wait_for_user_input(
            scenario_description=scenario_description,
            ds_trace_desc=ds_trace_desc,
            current_code=current_code,
            hypothesis_candidates=exp.hypothesis_candidates,
            target_hypothesis=target_hypothesis,
            target_hypothesis_index=target_hypothesis_index,
            task=exp.pending_tasks_list[0][0],
            exp=exp,
        )


class FBDSInteractor(DSInteractor):
    def dump_and_wait_for_user_input(
        self,
        scenario_description: str,
        ds_trace_desc: str,
        current_code: str,
        hypothesis_candidates: list[DSHypothesis],
        target_hypothesis: DSHypothesis,
        target_hypothesis_index: int,
        task: Task,
        exp: DSExperiment,
    ) -> DSExperiment:
        information_to_user = {
            "competition": DS_RD_SETTING.competition,
            "scenario_description": scenario_description,
            "ds_trace_desc": ds_trace_desc,
            "current_code": current_code,
            "hypothesis_candidates": hypothesis_candidates,
            "target_hypothesis": (
                hypothesis_candidates[target_hypothesis_index] if target_hypothesis_index != -1 else target_hypothesis
            ),
            "target_hypothesis_index": target_hypothesis_index,
            "task": task,
            "expired_datetime": datetime.now() + timedelta(seconds=DS_RD_SETTING.user_interaction_wait_seconds),
            "former_user_instructions": exp.user_instructions,
        }
        session_id = uuid.uuid4().hex
        DS_RD_SETTING.user_interaction_mid_folder.mkdir(parents=True, exist_ok=True)
        pickle.dump(information_to_user, open(DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}.pkl", "wb"))
        while (
            Path(DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}.pkl").exists()
            and pickle.load(open(DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}.pkl", "rb"))[
                "expired_datetime"
            ]
            > datetime.now()
            and not (DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}_RET.json").exists()
        ):
            time.sleep(5)
        Path(DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}.pkl").unlink(missing_ok=True)
        if not (DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}_RET.json").exists():
            return exp
        else:
            user_feedback = json.load(open(DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}_RET.json"))
            if user_feedback["action"] == "confirm":
                return exp
            elif user_feedback["action"] == "rewrite":
                exp.hypothesis.hypothesis = user_feedback["target_hypothesis"]
                exp.pending_tasks_list[0][0].description = user_feedback["task_description"]
                exp.set_user_instructions(user_feedback["user_instruction"])
                Path(DS_RD_SETTING.user_interaction_mid_folder / f"{session_id}_RET.json").unlink(missing_ok=True)
                return exp
