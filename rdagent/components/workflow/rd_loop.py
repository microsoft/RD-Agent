"""
Model workflow with session control
It is from `rdagent/app/qlib_rd_loop/model.py` and try to replace `rdagent/app/qlib_rd_loop/RDAgent.py`
"""

import asyncio
import json
from multiprocessing import Queue
from pathlib import Path
from typing import Any

from rdagent.components.workflow.conf import BasePropSetting
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.proposal import (
    Experiment2Feedback,
    ExperimentPlan,
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
    Trace,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.utils.qlib import ALPHA20, validate_qlib_features
from rdagent.utils.workflow import LoopBase, LoopMeta


class RDLoop(LoopBase, metaclass=LoopMeta):

    def __init__(self, PROP_SETTING: BasePropSetting):
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")
        self.hypothesis_gen: HypothesisGen = (
            import_class(PROP_SETTING.hypothesis_gen)(scen)
            if hasattr(PROP_SETTING, "hypothesis_gen") and PROP_SETTING.hypothesis_gen
            else None
        )

        self.plan: ExperimentPlan = {
            "features": ALPHA20,
            "feature_codes": {},
        }  # for user interaction

        self.hypothesis2experiment: Hypothesis2Experiment = (
            import_class(PROP_SETTING.hypothesis2experiment)()
            if hasattr(PROP_SETTING, "hypothesis2experiment") and PROP_SETTING.hypothesis2experiment
            else None
        )

        self.coder: Developer = (
            import_class(PROP_SETTING.coder)(scen) if hasattr(PROP_SETTING, "coder") and PROP_SETTING.coder else None
        )
        self.runner: Developer = (
            import_class(PROP_SETTING.runner)(scen) if hasattr(PROP_SETTING, "runner") and PROP_SETTING.runner else None
        )

        self.summarizer: Experiment2Feedback = (
            import_class(PROP_SETTING.summarizer)(scen)
            if hasattr(PROP_SETTING, "summarizer") and PROP_SETTING.summarizer
            else None
        )
        self.trace = Trace(scen=scen)
        super().__init__()

    # excluded steps
    def _set_interactor(self, user_request_q: Queue, user_response_q: Queue):
        self.user_request_q = user_request_q
        self.user_response_q = user_response_q

    def _init_base_features(self, base_features_path: str | None):
        if base_features_path is not None:
            try:
                base_dir = Path(base_features_path)
                base_factors_file = base_dir / "base_factors.json"

                feature_codes: dict[str, str] = {}
                for py_file in sorted(base_dir.glob("*.py")):
                    feature_codes[py_file.name] = py_file.read_text()
                self.plan["feature_codes"] = feature_codes

                if not base_factors_file.exists():
                    logger.info(f"No base_factors.json found under {base_dir}. Keeping default base features.")
                    logger.info(f"{len(feature_codes)} feature code files loaded from {base_dir}.")
                else:
                    with base_factors_file.open("r") as f:
                        features = json.load(f)

                    if not isinstance(features, dict):
                        raise ValueError(
                            "`base_factors.json` must contain a JSON object of feature_name -> expression."
                        )

                    if validate_qlib_features(list(features.values())):
                        self.plan["features"] = features
                        logger.info(
                            f"Loaded base features from {base_factors_file}. {len(features)} features loaded and {len(feature_codes)} feature code files loaded."
                        )
                    else:
                        logger.warning(
                            f"Base feature validation failed for features loaded from {base_factors_file}. Using default features."
                        )
            except Exception as e:
                logger.warning(f"Failed to load base features from {base_features_path}: {e}. Using default features.")
        else:
            logger.info("No base features path provided. Using default features.")

    def _interact_init_params(self) -> None:
        if not (hasattr(self, "user_request_q") and hasattr(self, "user_response_q")):
            return

        logger.info("Waiting for user interaction on initial parameters...")
        try:
            self.user_request_q.put(
                {
                    "user_instruction": None,
                }
            )
            res_dict = self.user_response_q.get()
            logger.info("Received user instruction response.")
            self.plan.update(res_dict)

            if "feature_codes" not in self.plan:
                self.plan[
                    "user_instruction"
                ] += f"\n\n{str(list(self.plan['feature_codes'].keys()))} has been configured as the base factor; do not generate duplicate factors."
            fea_valid_msg = ""
            while True:
                logger.info("Requesting base feature configuration from user.")
                self.user_request_q.put(
                    {
                        "features": self.plan["features"],
                        "feature_validation_msg": fea_valid_msg,
                    }
                )
                self.plan["features"] = self.user_response_q.get()
                logger.info("Received base feature configuration response.")
                if validate_qlib_features(list(self.plan["features"].values())):
                    logger.info(f"Base feature validation passed. {len(self.plan['features'])} features selected.")
                    break
                else:
                    logger.info("Base feature validation failed. Asking user to revise.")
                    fea_valid_msg = "Some features are invalid, please revise."

        except (EOFError, OSError):
            logger.info("User interaction failed, using default initial parameters.")
            return
        logger.info("Received user interaction on initial parameters.")

    def _interact_hypo(self, hypo: Hypothesis) -> Hypothesis:
        if not (hasattr(self, "user_request_q") and hasattr(self, "user_response_q")):
            return hypo

        logger.info("Waiting for user interaction on hypothesis...")
        try:
            self.user_request_q.put(hypo.__dict__)
            res_dict = self.user_response_q.get()
            modified_hypo = type(hypo)(**res_dict)
        except (EOFError, OSError, TypeError):
            logger.info("User interaction failed, using original hypothesis.")
            return hypo
        logger.info("Received user interaction on hypothesis.")
        return modified_hypo

    def _interact_feedback(self, feedback: HypothesisFeedback) -> HypothesisFeedback:
        if not (hasattr(self, "user_request_q") and hasattr(self, "user_response_q")):
            return feedback

        logger.info("Waiting for user interaction on feedback...")
        try:
            self.user_request_q.put(feedback.__dict__)
            res_dict = self.user_response_q.get()
            modified_feedback = HypothesisFeedback(**res_dict)
        except (EOFError, OSError, TypeError):
            logger.info("User interaction failed, using original feedback.")
            return feedback
        logger.info("Received user interaction on feedback.")
        return modified_feedback

    def _propose(self):
        hypothesis = self.hypothesis_gen.gen(self.trace, self.plan)

        # user can change the hypothesis here
        hypothesis = self._interact_hypo(hypothesis)

        logger.log_object(hypothesis, tag="hypothesis generation")
        return hypothesis

    def _exp_gen(self, hypothesis: Hypothesis):
        exp = self.hypothesis2experiment.convert(hypothesis, self.trace)
        logger.log_object(exp.sub_tasks, tag="experiment generation")
        return exp

    # included steps
    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                hypo = self._propose()
                exp = self._exp_gen(hypo)
                exp.base_features = self.plan["features"]
                exp.base_feature_codes = self.plan["feature_codes"]
                if exp.based_experiments:
                    exp.based_experiments[-1].base_features = self.plan["features"]
                    exp.based_experiments[-1].base_feature_codes = self.plan["feature_codes"]
                return {"propose": hypo, "exp_gen": exp}
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        exp = self.coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        # TODO: the logic branch of exception should be moved to summarizer
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            feedback = HypothesisFeedback(
                reason=str(e),
                decision=False,
                code_change_summary="",
                acceptable=False,
            )
        else:
            feedback = self.summarizer.generate_feedback(prev_out["running"], self.trace)
        feedback = self._interact_feedback(feedback)
        logger.log_object(feedback, tag="feedback")
        return feedback

    def record(self, prev_out: dict[str, Any]):
        feedback = prev_out["feedback"]
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen", {}).get("exp_gen")
        self.trace.sync_dag_parent_and_hist((exp, feedback), prev_out[self.LOOP_IDX_KEY])
