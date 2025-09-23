import json
import os
import pickle
import re
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fire
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.experiment import FBWorkspace
from rdagent.core.proposal import ExperimentFeedback, SOTAexpSelector, Trace
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log.storage import FileStorage
from rdagent.log.utils import extract_json
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T
from rdagent.utils.fmt import shrink_text
from rdagent.utils.workflow import wait_retry

# --- Configuration Constants ---
MAX_API_RETRIES = int(os.getenv("MAX_API_RETRIES", 5))
DEFAULT_NUM_WORKERS = int(os.getenv("DEFAULT_NUM_WORKERS", 2))
MAX_SOTA_CANDIDATES = int(os.getenv("MAX_SOTA_CANDIDATES", 6))

logger.add("selector.log")
# ==============================================================================
# ## SOTA Selector Implementations
# ==============================================================================


class GlobalSOTASelector(SOTAexpSelector):
    """
    Selects the single best State-Of-The-Art (SOTA) experiment from the entire trace history.
    """

    def __init__(self):
        logger.info("Using selector policy: GlobalSOTASelector")

    def get_sota_exp_to_submit(self, trace: Trace, **kwargs) -> DSExperiment | None:
        """
        Returns the single best experiment from all historical runs.
        """
        return trace.sota_experiment(search_type="all")


class AutoSOTAexpSelector(SOTAexpSelector):
    """
    Uses an LLM to select the best SOTA experiment from a list of candidates.
    Candidates are retrieved from the leaves of the experiment trace tree.
    """

    def __init__(self):
        logger.info("Using selector policy: AutoSOTAexpSelector")

    @wait_retry(retry_n=MAX_API_RETRIES)
    def get_sota_exp_to_submit(self, trace: Trace, **kwargs) -> DSExperiment | None:
        """
        Retrieves SOTA experiments, then uses an LLM to choose the most promising one.
        """
        sota_exp_fb_list = self.collect_sota_candidates(trace)

        if not sota_exp_fb_list:
            logger.info("AutoSOTASelector: No SOTA experiments found in trace.")
            return None

        if len(sota_exp_fb_list) == 1:
            logger.info("AutoSOTASelector: Only one SOTA candidate found, selecting it.")
            return sota_exp_fb_list[0][0]

        logger.info(f"AutoSOTASelector: {len(sota_exp_fb_list)} SOTA candidates found. Querying LLM for selection.")

        # Build prompt for LLM
        sota_prompt_text = "Historical SOTA experiments:\n\n"
        system_prompt = T(".prompts:auto_sota_selector.system").r(scenario=trace.scen.get_scenario_all_desc())
        for i, (exp, _) in enumerate(sota_exp_fb_list):
            if exp and exp.result is not None:
                current_final_score = pd.DataFrame(exp.result).loc["ensemble"].iloc[0]
                desc = T("scenarios.data_science.share:describe.exp").r(exp=exp)
                new_experiment_content = f"""SOTA experiment No. {i+1}:
                        Description: {desc}
                        Final score: {current_final_score}\n\n"""

                temp_user_prompt = T(".prompts:auto_sota_selector.user").r(
                    historical_sota_exp_with_desc_and_scores=sota_prompt_text + new_experiment_content,
                )

                token_size = APIBackend().build_messages_and_calculate_token(
                    user_prompt=temp_user_prompt,
                    system_prompt=system_prompt,
                )
                if token_size >= LLM_SETTINGS.chat_token_limit:
                    logger.warning(f"Token limit reached at experiment {i+1}. Stopping.")
                    break

                sota_prompt_text += new_experiment_content

        # Query LLM
        user_prompt = T(".prompts:auto_sota_selector.user").r(historical_sota_exp_with_desc_and_scores=sota_prompt_text)

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            json_target_type=Dict[str, Any],
        )
        response_dict = json.loads(response)
        selected_idx = response_dict.get("selected_SOTA_idx")

        # Process LLM response
        if selected_idx and isinstance(selected_idx, int) and 0 < selected_idx <= len(sota_exp_fb_list):
            sota_submit = sota_exp_fb_list[selected_idx - 1][0]
            logger.info(f"AutoSOTASelector: LLM selected experiment No. {selected_idx}.")
            return sota_submit

        logger.warning("AutoSOTASelector: LLM selection was invalid. Falling back to the latest SOTA experiment.")
        return sota_exp_fb_list[-1][0] if sota_exp_fb_list else None

    def collect_sota_candidates(self, trace: Trace) -> list:
        """Helper to gather SOTA experiments from trace leaves."""
        leaves = trace.get_leaves()
        if len(leaves) < 2:
            return trace.experiment_and_feedback_list_after_init(
                return_type="sota", search_type="all", max_retrieve_num=DS_RD_SETTING.max_sota_retrieved_num
            )

        logger.info(f"AutoSOTASelector: {len(leaves)} branches found, collecting SOTA from each.")
        all_sota_candidates = []
        num_per_trace = max(DS_RD_SETTING.max_sota_retrieved_num // len(leaves), 2)

        for leaf in leaves:
            sota_from_branch = trace.experiment_and_feedback_list_after_init(
                return_type="sota", search_type="ancestors", selection=(leaf,), max_retrieve_num=num_per_trace
            )
            all_sota_candidates.extend(sota_from_branch)

        # Remove duplicates and limit total number of candidates
        unique_sota_list = list(set(all_sota_candidates))
        is_higher_better = trace.scen.metric_direction
        unique_sota_list.sort(
            key=lambda exp_fb: pd.DataFrame(exp_fb[0].result).loc["ensemble"].iloc[0],
            reverse=is_higher_better,
        )
        return unique_sota_list[: DS_RD_SETTING.max_sota_retrieved_num]


class BestValidSelector(SOTAexpSelector):
    """
    Selects the top N experiments based on their performance score.
    Can operate across the entire trace or on a per-branch basis.
    """

    def __init__(self, num_candidates: int = 1, use_decision: bool = True, each_trace: bool = False):
        """
        Args:
            num_candidates (int): The number of top experiments to return.
            use_decision (bool): If True, filters out experiments marked with a negative decision.
            each_trace (bool): If True, selects top candidates from each branch instead of globally.
        """
        logger.info(
            f"Using selector policy: BestValidSelector (num_candidates={num_candidates}, each_trace={each_trace})"
        )
        self.num_candidates = num_candidates
        self.use_decision = use_decision
        self.each_trace = each_trace

    def get_sota_exp_to_submit(self, trace: Trace, **kwargs) -> DSExperiment | None:
        """
        Sorts all valid experiments by score and returns the top N.
        """
        top_experiments = self.collect_sota_candidates(trace)
        if top_experiments:
            return top_experiments[0]
        return None

    def collect_sota_candidates(self, trace: Trace) -> list[DSExperiment] | None:
        """Helper to gather SOTA experiments from trace leaves."""
        """
        Sorts all valid experiments by score and returns the top N.
        """
        direction_sign = 1 if trace.scen.metric_direction else -1

        def get_sort_key(exp_fb: Tuple[DSExperiment, ExperimentFeedback]) -> Tuple[bool, float]:
            exp, feedback = exp_fb
            score = -np.inf
            if exp.result is not None:
                try:
                    score = pd.DataFrame(exp.result).loc["ensemble"].iloc[0]
                    if isinstance(score, str):
                        score = float(score.strip("tensor()"))
                    score = direction_sign * score
                except:
                    logger.warning(f"Failed to extract score from result {exp.result}")

            # Sort key prioritizes decision (True > False), then score
            return (feedback.decision, score) if self.use_decision else score

        def get_sort_key_without_decision(exp_fb: Tuple[DSExperiment, ExperimentFeedback]) -> Tuple[bool, float]:
            exp, feedback = exp_fb
            score = -np.inf
            if exp.result is not None:
                try:
                    score = pd.DataFrame(exp.result).loc["ensemble"].iloc[0]
                    if isinstance(score, str):
                        score = float(score.strip("tensor()"))
                    score = direction_sign * score
                except:
                    logger.warning(f"Failed to extract score from result {exp.result}")

            return score

        # Collect candidates
        if self.each_trace:
            # Add best experiment without decision
            hist = trace.hist.copy()
            hist.sort(key=get_sort_key_without_decision, reverse=True)
            candidate_list = [hist[0]]

            root_to_experiments = {}
            for node in range(len(trace.hist)):
                parents = trace.get_parents(node)
                if parents:
                    root = parents[0]
                    if root not in root_to_experiments:
                        root_to_experiments[root] = []
                    root_to_experiments[root].append(trace.hist[node])

            # Select top-k from each branch
            num_per_leaf = max(self.num_candidates // len(root_to_experiments), 2)
            for root, exps in root_to_experiments.items():
                if not exps:
                    continue
                exps.sort(key=get_sort_key, reverse=True)
                candidate_list.extend(exps[:num_per_leaf])
            # Remove duplicates
            candidate_list = list(set(candidate_list))
        else:
            candidate_list = trace.experiment_and_feedback_list_after_init(return_type="all", search_type="all")

        if not candidate_list:
            logger.info("BestValidSelector: No experiments found in trace.")
            return None

        # Sort and select the top N
        candidate_list.sort(key=get_sort_key_without_decision, reverse=True)

        top_experiments = [exp for exp, _ in candidate_list[: self.num_candidates]]
        logger.info(f"BestValidSelector: Selected {len(top_experiments)} experiments.")
        return top_experiments


class ValidationSelector(SOTAexpSelector):
    """
    A meta-selector that re-validates candidates from a base selector.

    It then generates a consistent validation dataset and grading script,
    re-runs all candidates on this new data, and returns the best performer.
    """

    def __init__(
        self,
        candidate: List[Tuple[DSExperiment, str]],
        direction_sign: int,
        competition: str,
        only_sample: bool,
        sample_code_path: str,
        sample_rate: float = 0.8,
    ):
        self.candidate = candidate
        self.direction_sign = direction_sign
        self.competition = competition
        self.only_sample = only_sample
        self.sample_code_path = Path(sample_code_path)
        self.sample_rate = sample_rate
        self.hypothesis_loop_id = {exp.hypothesis.hypothesis: loop_id for exp, loop_id in self.candidate}
        self.hypothesis_exp = {exp.hypothesis.hypothesis: exp for exp, loop_id in self.candidate}

    def get_sota_exp_to_submit(self, trace: Trace) -> DSExperiment | None:
        """Helper to gather SOTA experiments from trace leaves."""
        """
        Sorts all valid experiments by score and returns the top N.
        """

        mock_folder = f"/tmp/mock/{self.competition}"

        try:
            data_py_code, grade_py_code = self._prepare_validation_scripts(
                reference_exp=self.candidate[0][0], competition=self.competition, mock_folder=mock_folder
            )
        except RuntimeError as e:
            logger.error(f"ValidationSelector: Failed to prepare validation environment. {e}")
            shutil.rmtree(mock_folder, ignore_errors=True)
            return None

        validation_tasks = [
            (process_experiment, (exp, self.competition, mock_folder, grade_py_code, loop_id))
            for exp, loop_id in self.candidate
        ]
        results = multiprocessing_wrapper(validation_tasks, n=min(DEFAULT_NUM_WORKERS, (len(self.candidate) + 1) // 2))

        if not results:
            logger.warning("ValidationSelector: Validation run produced no results.")
            return None

        # 4. Process results and select the best one
        valid_results = [
            (
                self.hypothesis_exp.get(exp.hypothesis.hypothesis),
                self.hypothesis_loop_id.get(exp.hypothesis.hypothesis),
                valid_score,
                test_score,
            )
            for exp, valid_score, test_score in results
            if test_score is not None
        ]
        if not valid_results:
            logger.warning("ValidationSelector: No candidates scored successfully during validation.")
            return None

        valid_results.sort(key=lambda x: (x[3]) * self.direction_sign, reverse=True)
        best_exp, best_loop_id = valid_results[0][0], valid_results[0][1]

        for loop_id, valid_score, test_score in [(i[1], i[2], i[3]) for i in valid_results]:
            logger.info(f"ValidationSelector: Loop_id={loop_id} ->valid score={valid_score}, test score={test_score}")
        logger.info(
            f"ValidationSelector: Best experiment from validation is loop_id={best_loop_id} with valid score={valid_results[0][2]}, test score={valid_results[0][3]}"
        )
        if len(valid_results) <= 1 or valid_results[0][3] == valid_results[-1][3]:
            logger.warning(f"ValidationSelector: There aren't enough scores to compare, current: {len(valid_results)}.")
            return None

        return best_exp

    def print_code(self, data_py_code: str, grade_py_code: str):
        logger.info("Successfully ran data.py.")
        print("======== data.py ========")
        print(data_py_code)
        print("======== grade.py ========")
        print(grade_py_code)
        print("======== code end ========")

    def _prepare_validation_scripts(
        self, reference_exp: DSExperiment, competition: str, mock_folder: str
    ) -> Tuple[str, str]:
        """Generates and verifies data.py and grade.py using an LLM."""
        input_folder = T("scenarios.data_science.share:scen.input_path").r()
        mock_input_path = Path(mock_folder) / input_folder
        mock_input_path.mkdir(parents=True, exist_ok=True)

        data_py_path = Path(mock_folder) / "data.py"
        grade_py_path = Path(mock_folder) / "grade.py"
        label_path = Path(mock_folder) / "workspace_input/label.csv"
        reference_code = reference_exp.experiment_workspace.file_dict.get("main.py", "")
        if not reference_code:
            raise RuntimeError("ValidationSelector: No code found in the reference experiment.")

        if (self.sample_code_path / competition / "data.py").exists():
            shutil.copy(self.sample_code_path / competition / "data.py", data_py_path)
            shutil.copy(self.sample_code_path / competition / "grade.py", grade_py_path)
            data_py_code = data_py_path.read_text()
            grade_py_code = grade_py_path.read_text()
            if not label_path.exists():
                ws = FBWorkspace()
                if self.sample_rate != 0.8:
                    data_py_code = data_py_code.replace("0.8", str(self.sample_rate)).replace(
                        "0.2", str(round(1 - self.sample_rate, 2))
                    )
                ws.inject_code_from_file_dict(reference_exp.experiment_workspace)
                ws.inject_files(**{f"data.py": data_py_code})
                env = get_ds_env(
                    extra_volumes={
                        str(Path(mock_folder) / input_folder): {"bind": input_folder, "mode": "rw"},
                        f"{DS_RD_SETTING.local_data_path}/{competition}": "./source",
                    },
                    running_timeout_period=DS_RD_SETTING.full_timeout,
                )
                result = ws.run(
                    env=env, entry=f"python data.py --cache-buster={time.time()}"
                )  # Do not cache the result
                if result.exit_code == 0:
                    self.print_code(data_py_code, grade_py_code)
            return data_py_code, grade_py_code

        # --- Generate data.py if needed ---
        if not data_py_path.exists() or not label_path.exists():
            logger.info(f"Generating synthetic data script: {data_py_path}")
            data_py_code = self._generate_and_run_script(
                script_type="data",
                prompt_template_key="sample_data",
                reference_exp=reference_exp,
                competition=competition,
                mock_folder=mock_folder,
                prompt_kwargs={"reference_code": reference_code, "input_folder": input_folder},
            )
            data_py_path.write_text(data_py_code)

        data_py_code = data_py_path.read_text()

        # --- Generate grade.py if needed ---
        if not grade_py_path.exists():
            logger.info(f"Generating grading script: {grade_py_path}")
            grade_py_code = self._generate_and_run_script(
                script_type="grade",
                prompt_template_key="grade",
                reference_exp=reference_exp,
                competition=competition,
                mock_folder=mock_folder,
                prompt_kwargs={
                    "reference_code": reference_code,
                    "sample_code": data_py_code,
                    "input_folder": input_folder,
                },
            )
            grade_py_path.write_text(grade_py_code)
            self.print_code(data_py_code, grade_py_code)
        return data_py_code, grade_py_path.read_text()

    def _generate_and_run_script(
        self,
        script_type: str,
        prompt_template_key: str,
        reference_exp: DSExperiment,
        competition: str,
        mock_folder: str,
        prompt_kwargs: dict,
    ) -> str:
        """A helper to generate, run, and validate a script (data.py or grade.py)."""
        system_prompt = T(".prompts:sample_data.system").r()  # Generic system prompt for both
        input_folder = T("scenarios.data_science.share:scen.input_path").r()

        err_msg = ""
        for _ in range(MAX_API_RETRIES):
            user_prompt = T(f".prompts:{prompt_template_key}.user").r(error=err_msg, **prompt_kwargs)

            generated_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt, system_prompt=system_prompt
                )
            )

            # Create a temporary workspace to test the generated script
            ws = FBWorkspace()
            ws.inject_code_from_file_dict(reference_exp.experiment_workspace)
            ws.inject_files(**{f"{script_type}.py": generated_code})
            reference_code = reference_exp.experiment_workspace.file_dict.get("main.py", "")
            ws.inject_files(**{"reference_code.py": reference_code})

            if script_type == "data":
                # For data.py, we need the original data to sample from
                env = get_ds_env(
                    extra_volumes={
                        str(Path(mock_folder) / input_folder): {"bind": input_folder, "mode": "rw"},
                        f"{DS_RD_SETTING.local_data_path}/{competition}": "./source",
                    },
                    running_timeout_period=DS_RD_SETTING.full_timeout,
                )
            else:  # For grade.py, we only need the generated data
                shutil.copy(
                    str(Path(mock_folder) / "submission.csv"),
                    str(ws.workspace_path / "submission.csv"),
                )
                env = get_ds_env(
                    extra_volumes={str(Path(mock_folder) / input_folder): {"bind": input_folder, "mode": "rw"}}
                )

            result = ws.run(
                env=env, entry=f"python {script_type}.py --cache-buster={time.time()}"
            )  # Do not cache the result
            stdout = re.sub(r"^chmod:.*\n?", "", result.get_truncated_stdout(), flags=re.MULTILINE)

            if result.exit_code == 0:
                logger.info(f"Successfully generated and ran {script_type}.py.")
                if script_type == "data":
                    env = get_ds_env(
                        extra_volumes={str(Path(mock_folder) / input_folder): {"bind": input_folder, "mode": "rw"}},
                        running_timeout_period=DS_RD_SETTING.full_timeout,
                    )
                    result = ws.run(env=env, entry=f"python reference_code.py")
                    stdout = re.sub(r"^chmod:.*\n?", "", result.get_truncated_stdout(), flags=re.MULTILINE)
                    if result.exit_code == 0:
                        # move submission.csv to mock_folder
                        if Path(ws.workspace_path / "submission.csv").exists():
                            shutil.copy(
                                str(ws.workspace_path / "submission.csv"),
                                str(Path(mock_folder) / "submission.csv"),
                            )
                            return generated_code
                        else:
                            err_msg = "No submission.csv found in workspace after running main.py with generated data."
                    else:
                        err_msg = f"Error in main.py with generated data: {shrink_text(stdout, context_lines=20, line_len=500)}"
                else:
                    score = _parsing_score(stdout)
                    if score is not None:
                        return generated_code
                    else:
                        err_msg = f"No score found in stdout: {stdout}."
            else:
                err_msg = f"Error in {script_type}.py: {shrink_text(stdout, context_lines=20, line_len=500)}"

            logger.warning(f"Attempt to generate {script_type}.py failed. Retrying... Error: {err_msg}")
        raise RuntimeError(f"Failed to generate a working {script_type}.py after {MAX_API_RETRIES} attempts.")


# ==============================================================================
# ## Worker and Utility Functions
# ==============================================================================


def process_experiment(
    exp: DSExperiment, competition: str, folder: str, grade_py_code: str, loop_id: str
) -> Tuple[DSExperiment, Optional[float], Optional[float]]:
    """
    Worker function to process a single experiment in an isolated directory.
    This function is designed to be called by a multiprocessing pool.
    """
    if loop_id is None:
        logger.error("Could not find loop_id for a given experiment.")
        loop_id = "unknown"

    input_folder = T("scenarios.data_science.share:scen.input_path").r()
    valid_score = None

    try:
        ws = FBWorkspace()
        logger.info(f"Experiment files: {exp.experiment_workspace.file_dict.keys()}")
        ws.inject_code_from_file_dict(exp.experiment_workspace)

        # Run main script
        env = get_ds_env(
            extra_volumes={f"/tmp/mock/{competition}/{input_folder}": input_folder},
            running_timeout_period=DS_RD_SETTING.full_timeout,
        )
        result = ws.run(env=env, entry="python main.py")
        execute_ret_code = result.exit_code
        logger.info(f"Ran {competition}/{loop_id}/main.py; exit_code: {execute_ret_code}")

        # Run grading script if main script succeeded
        grade_stdout = ""
        if execute_ret_code == 0:
            score_fp = ws.workspace_path / "scores.csv"
            if score_fp.exists():
                try:
                    valid_score = pd.read_csv(score_fp, index_col=0).loc["ensemble"].iloc[0]
                except Exception as e:
                    logger.error(f"Error parsing valid score from {score_fp}: {e}")
            ws.inject_files(**{"grade.py": grade_py_code})
            env.conf.running_timeout_period = DS_RD_SETTING.debug_timeout
            result = ws.run(env=env, entry="python grade.py")
            if result.exit_code == 0:
                grade_stdout = re.sub(r"^chmod:.*\n?", "", result.get_truncated_stdout(), flags=re.MULTILINE)
            logger.info(f"Ran grade.py for {competition}/{loop_id}; exit_code: {result.exit_code}")
        else:
            logger.warning(f"Skipping grading for {competition}/{loop_id} due to main.py execution failure.")

    except Exception as e:
        logger.error(f"CRITICAL ERROR while processing experiment {competition}/{loop_id}: {e}")
        return exp, None, None

    # Score parsing
    return exp, valid_score, _parsing_score(grade_stdout)


def _parsing_score(grade_stdout: str) -> Optional[float]:
    for line in grade_stdout.splitlines():
        line = line.strip()
        if "score" not in line:
            continue
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", line)
        if not m:
            continue
        json_str = m.group(0)
        try:
            # Priority 1: JSON parsing
            return float(json.loads(json_str)["score"])
        except:
            pass
        try:
            # Priority 2: Eval dict
            return float(eval(json_str)["score"])
        except:
            pass
        try:
            # Priority 3: Regex for the last number in the string
            return float(re.findall(r"[-+]?\d*\.\d+|\d+", json_str)[-1])
        except:
            pass
    return None


def check_hit(selected_exp: DSExperiment, trace: Trace, sota_result: Dict[str, Any]) -> bool:
    """Checks if any of the selected experiments are considered medal-winning."""
    if not selected_exp:
        return False

    index = trace.exp2idx(selected_exp)
    # Check by loop_id if available
    if hasattr(trace, "idx2loop_id"):
        loop_id = trace.idx2loop_id.get(index)
        if loop_id:
            if loop_id in sota_result.get("medal_loops", []):
                return True
            return False

    # Fallback to checking by index
    if index in sota_result.get("medal_loops_index", []):
        return True
    return False


def try_get_loop_id(trace: Trace, exp: DSExperiment):
    index = trace.exp2idx(exp)
    if hasattr(trace, "idx2loop_id"):
        return trace.idx2loop_id.get(index)
    return index


def extract_tar(tar_path: str, to_dir: str = "log") -> str:
    with tarfile.open(tar_path, mode="r:*") as tar:
        tar.extractall(path=to_dir)


# ==============================================================================
# ## Main Orchestration Logic
# ==============================================================================


def evaluate_one_trace(
    selector_name: str,
    trace: Trace,
    debug: bool,
    only_sample: bool,
    sample_code_path: str,
    sota_result: dict[str, Any] = {},
    experiment: str = "validation",
    log_path: Path | None = None,
    sample_rate: float = 0.8,
) -> Tuple[str, bool, str]:
    """
    Loads a single trace, uses the specified selector to pick an experiment,
    and checks if the selection was a "hit" (a known SOTA solution).
    """
    competition = trace.scen.competition
    hit = False
    sota_exp_stat = ""

    # Example of scenario-specific adjustment
    if competition == "detecting-insults-in-social-commentary":
        trace.scen.metric_direction = 1
    direction_sign = 1 if trace.scen.metric_direction else -1

    # --- Selector Instantiation ---
    # The core logic is now encapsulated in these selectors.
    if selector_name == "global":
        selector = GlobalSOTASelector()
    elif selector_name == "auto":
        selector = AutoSOTAexpSelector()
    elif selector_name == "best_valid":
        # These params can be configured or passed via CLI
        selector = BestValidSelector(num_candidates=1, use_decision=True, each_trace=False)

    if selector_name == "validation":
        if not Path(f"{DS_RD_SETTING.local_data_path}/{competition}").exists():
            logger.warning(f"Competition {DS_RD_SETTING.local_data_path}/{competition} does not exist, skipping.")
            return competition, False, sota_exp_stat
        # The ValidationSelector is used to select the best re-test score.
        quick_selector = BestValidSelector(num_candidates=1, use_decision=True, each_trace=False)
        quick_selected_exps = quick_selector.get_sota_exp_to_submit(trace)
        if debug:
            quick_hit = check_hit(quick_selected_exps, trace, sota_result)
            logger.info(f"BestvalidSelector for {experiment} - {competition}: {'HIT' if quick_hit else 'MISS'}")

        base_selector = BestValidSelector(num_candidates=MAX_SOTA_CANDIDATES, use_decision=True, each_trace=True)
        candidate_exps = base_selector.collect_sota_candidates(trace)
        if not candidate_exps:
            logger.info("ValidationSelector: Base selector returned no candidates.")
            return competition, False, sota_exp_stat

        logger.info(f"ValidationSelector: Received {len(candidate_exps)} candidates for validation.")
        pool_hit = False
        if debug:
            pool_hit = any(check_hit(candidate_exp, trace, sota_result) for candidate_exp in candidate_exps)
        else:
            for exp in candidate_exps:
                loop_id = try_get_loop_id(trace, exp)
                sota_mle_score_paths = [i for i in log_path.rglob(f"Loop_{loop_id}/running/mle_score/**/*.pkl")]
                if len(sota_mle_score_paths):
                    with sota_mle_score_paths[0].open("rb") as f:
                        sota_mle_score = extract_json(pickle.load(f))
                        if sota_mle_score.get("any_medal", False):
                            pool_hit = True
                            break
        if not pool_hit:
            logger.info("ValidationSelector: Selector's candidates did not hit any medal. Skipping validation.")
            return competition, False, sota_exp_stat

        selector = ValidationSelector(
            candidate=[(exp, try_get_loop_id(trace, exp)) for exp in candidate_exps],
            direction_sign=direction_sign,
            competition=competition,
            only_sample=only_sample,
            sample_code_path=sample_code_path,
            sample_rate=sample_rate,
        )

    selected_sota_exps = selector.get_sota_exp_to_submit(trace)
    if selector_name == "validation" and selected_sota_exps is None:
        selected_sota_exps = quick_selected_exps

    # --- Run Selection and Check for Hit ---
    logger.info(f"Running selector '{selector_name}' on trace for competition '{competition}'...")
    if debug:
        hit = check_hit(selected_sota_exps, trace, sota_result)
        logger.info(f"Result for {experiment} - {competition}: {'HIT' if hit else 'MISS'}")
    elif selector_name == "validation":
        loop_id = selector.hypothesis_loop_id.get(selected_sota_exps.hypothesis.hypothesis)
        logger.info(f"Selected loop for {experiment} - {competition}: {loop_id=}")
        sota_mle_score_paths = [i for i in log_path.rglob(f"Loop_{loop_id}/running/mle_score/**/*.pkl")]
        if len(sota_mle_score_paths):
            with sota_mle_score_paths[0].open("rb") as f:
                sota_mle_score = extract_json(pickle.load(f))
                hit = sota_mle_score.get("any_medal", False)
                if hit:
                    if sota_mle_score["gold_medal"]:
                        sota_exp_stat = "gold"
                    elif sota_mle_score["silver_medal"]:
                        sota_exp_stat = "silver"
                    elif sota_mle_score["bronze_medal"]:
                        sota_exp_stat = "bronze"
    return competition, hit, sota_exp_stat


def select_on_existing_trace(
    selector_name: str,
    trace_root: str = "",
    experiment: str | None = None,
    competition: str | None = None,
    debug: bool = False,
    only_sample: bool = False,
    sample_code_path: str = "",
    sample_rate: float = 0.8,
):
    """
    Offline evaluation of a SOTA experiment selector on existing traces.

    Args:
        selector_name (str): Name of the selector to use. Options: 'global', 'auto', 'best_valid', 'validation'.
        trace_root (str): Path to the root directory containing trace folders.
        experiment (str | None): Name of the experiment to evaluate, e.g., "devoted-burro+massive-perch".
        competition (str | None): Name of the competition to evaluate, e.g., "detecting-insults-in-social-commentary".
        debug (bool): If True, debug mode.
        only_sample (bool): If True, only generates the sample code.
        sample_code_path (str): Path to the sample code.
    """
    result_dict = {}
    trace_root_path = Path(trace_root)

    # Prepare list of tasks for multiprocessing
    tasks = []
    if debug and experiment and "yaml" in trace_root:
        job_info = yaml.safe_load(open(str(Path(trace_root) / f"{experiment}.yaml"), "r"))
        if not competition:
            competition = os.getenv("DS_COMPETITION")
        for job in job_info:
            if job["submit_args"]["env"]["DS_COMPETITION"] == competition:
                tar_file = Path("/mnt/output") / job["results_dir"] / job["submit_args"]["env"]["RD_RES_NAME"]
                extract_tar(tar_file)
                debug = False

    if debug:
        for trace_folder in trace_root_path.iterdir():
            if not trace_folder.is_dir():
                continue
            if isinstance(experiment, str) and experiment:
                if trace_folder.name not in experiment:
                    continue
            for trace_pkl_path in trace_folder.glob("*.pkl"):
                if competition is not None and not competition in str(trace_pkl_path):
                    continue
                sota_result = {}
                trace = pickle.load(trace_pkl_path.open("rb"))
                try:
                    sota_loops_file = trace_folder / f"{trace_pkl_path.stem.split('_')[0]}_loops.json"
                    with open(sota_loops_file, "r") as f:
                        sota_result = json.load(f)
                except FileNotFoundError:
                    logger.warning(f"Could not find SOTA loops file for {trace.scen.competition}, skipping.")
                    continue

                if not sota_result.get("medal_loops"):
                    logger.info(f"No Medal loops defined for {trace.scen.competition}, skipping.")
                    continue

                tasks.append(
                    (
                        evaluate_one_trace,
                        (
                            selector_name,
                            trace,
                            debug,
                            only_sample,
                            sample_code_path,
                            sota_result,
                            trace_pkl_path.parent.name,
                            None,
                            sample_rate,
                        ),
                    )
                )
    else:
        log_path = next(
            d for d in Path("log").iterdir() if d.is_dir() and d.name != "pickle_cache" and not d.name.startswith("20")
        )
        logger.info(f"Loading trace from {log_path}")
        log_storage = FileStorage(log_path)
        all_traces = list(log_storage.iter_msg(tag="trace"))
        if not all_traces:
            logger.error("No valid trace found in log directory.")
            return

        trace = all_traces[-1].content
        tasks.append(
            (
                evaluate_one_trace,
                (selector_name, trace, debug, only_sample, sample_code_path, {}, "validation", log_path, sample_rate),
            )
        )

    if not tasks:
        logger.error(f"No .pkl trace files found in subdirectories of {trace_root}")
        return

    # Run evaluation in parallel
    hit_list = multiprocessing_wrapper(tasks, n=1)  # n=1 for sequential debugging, increase for parallel runs

    # Aggregate and report results
    hit_count = sum(hit for _, hit, _ in hit_list if hit is not None)
    total_valid_traces = len(hit_list)

    print("\n" + "=" * 50)
    print(f"Evaluation Summary for Selector: '{selector_name}'")
    print(f"Total Traces Processed: {total_valid_traces}")
    print(f"Total Hits: {hit_count}")
    if not debug and hit_count:
        print(f"Medal info: {hit_list[0][2]}")
    if total_valid_traces > 0:
        hit_rate = (hit_count / total_valid_traces) * 100
        print(f"Hit Rate: {hit_rate:.2f}%")
    print("=" * 50 + "\n")

    result_dict["summary"] = {
        "hit": hit_count,
        "total": total_valid_traces,
        "hit_rate": hit_rate if total_valid_traces > 0 else 0,
    }
    result_dict["details"] = [{comp: hit} for comp, hit, _ in hit_list]

    with open(f"result_{selector_name}.json", "w") as f:
        json.dump(result_dict, f, indent=4)
    logger.info(f"Results saved to result_{selector_name}.json")
    if "yaml" in trace_root and Path("log/log").exists():
        shutil.rmtree("log/log")


if __name__ == "__main__":
    fire.Fire(select_on_existing_trace)
