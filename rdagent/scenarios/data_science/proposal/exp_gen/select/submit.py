import json
import pickle
import random
import re
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, Tuple

import fire
import numpy as np
import pandas as pd
from loguru import logger

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.experiment import FBWorkspace
from rdagent.core.proposal import ExperimentFeedback, SOTAexpSelector, Trace
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.utils.agent.ret import PythonAgentOut
from rdagent.utils.agent.tpl import T
from rdagent.utils.fmt import shrink_text
from rdagent.utils.workflow import wait_retry


class GlobalSOTASelector(SOTAexpSelector):
    """
    return the latest SOTA experiment from the trace to submit
    """

    def __init__(
        self,
    ):
        print(f"Using global SOTA policy by default")

    def get_sota_exp_to_submit(self, trace: Trace) -> DSExperiment | None:

        return trace.sota_experiment(search_type="all")


class AutoSOTAexpSelector(SOTAexpSelector):
    """
    retrieve a list of SOTA experiments from the trace, then call the LLM to select the best one
    """

    def __init__(
        self,
    ):
        print(f"Using auto SOTA policy")

    @wait_retry(retry_n=5)
    def get_sota_exp_to_submit(self, trace: Trace) -> DSExperiment | None:
        # retrieve all SOTA experiments from the trace

        sota_exp_fb_list = trace.experiment_and_feedback_list_after_init(
            return_type="sota", search_type="all", max_retrieve_num=DS_RD_SETTING.max_sota_retrieved_num
        )
        logger.info(f"Auto SOTA selector: Found {len(sota_exp_fb_list)} SOTA experiments")
        if len(sota_exp_fb_list) == 0:
            logger.info("Auto SOTA selector: No SOTA in trace yet")
            return None

        elif len(sota_exp_fb_list) == 1:
            sota_idx_in_trace = trace.hist.index(sota_exp_fb_list[0])
            logger.info(
                f"Auto SOTA selector: Only one SOTA in trace, using it, which is the No. {sota_idx_in_trace + 1} in the trace"
            )
            return sota_exp_fb_list[0][0]

        else:
            logger.info(
                f"Auto SOTA selector: Multiple SOTA in trace, calling LLM to select the best one in {DS_RD_SETTING.max_sota_retrieved_num} SOTA experiments"
            )

            SOAT_exp_with_desc_and_scores = "Historical SOTA experiments:\n\n"

            leaves: list[int] = trace.get_leaves()

            if len(leaves) >= 2:

                logger.info(
                    f"Auto SOTA selector: {len(leaves)} traces found, collecting SOTA experiments from each trace"
                )
                # multiple trace case, collect the latest SOTA experiments from each trace
                new_sota_exp_fb_list: list[tuple[DSExperiment, ExperimentFeedback]] = []
                # calculate the number of SOTA experiments to retrieve from each trace, prevent it from becoming zero
                max_sota_retrieved_num_per_trace = max(DS_RD_SETTING.max_sota_retrieved_num // len(leaves), 2)
                # recall, due to the integer division, the final number of SOTA experiments to retrieve may be different
                for leaf in leaves:
                    sota_exp_fb_list_per_trace = trace.experiment_and_feedback_list_after_init(
                        return_type="sota",
                        search_type="ancestors",
                        selection=(leaf,),
                        max_retrieve_num=max_sota_retrieved_num_per_trace,
                    )
                    logger.info(
                        f"Auto SOTA selector: Collected {len(sota_exp_fb_list_per_trace)} SOTA experiments from trace with leaf #. {leaf}"
                    )

                    new_sota_exp_fb_list.extend(sota_exp_fb_list_per_trace)

                sota_exp_fb_list = list(set(new_sota_exp_fb_list))

                if len(sota_exp_fb_list) == 0:
                    logger.info("Auto SOTA selector: No SOTA in trace yet")
                    return None

                elif len(sota_exp_fb_list) == 1:
                    logger.info("Auto SOTA selector: Only one SOTA in trace, using it")
                    return sota_exp_fb_list[0][0]
                else:
                    logger.info(
                        f"Auto SOTA selector: select {len(sota_exp_fb_list)} of {len(new_sota_exp_fb_list)} SOTA experiments found in all traces, calling LLM to select the best one"
                    )
                    if len(sota_exp_fb_list) > DS_RD_SETTING.max_sota_retrieved_num:
                        sota_exp_fb_list = sorted(
                            sota_exp_fb_list,
                            key=lambda exp_fb: pd.DataFrame(exp_fb[0].result).loc["ensemble"].iloc[0],
                            reverse=not trace.scen.metric_direction,
                        )[-DS_RD_SETTING.max_sota_retrieved_num :]

            for i, (exp, ef) in enumerate(sota_exp_fb_list):
                if exp:
                    current_final_score = pd.DataFrame(exp.result).loc["ensemble"].iloc[0]
                    desc = T("scenarios.data_science.share:describe.exp").r(
                        exp=exp, heading="SOTA of previous exploration of the scenario"
                    )
                    SOAT_exp_with_desc_and_scores += f"""SOTA experiment No. {i+1}:
                        Description: {desc}
                        Final score: {current_final_score}\n\n"""

            system_prompt = T(".prompts:auto_sota_selector.system").r(scenario=trace.scen.get_scenario_all_desc())

            user_prompt = T(".prompts:auto_sota_selector.user").r(
                historical_sota_exp_with_desc_and_scores=SOAT_exp_with_desc_and_scores,
            )

            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_target_type=Dict[str, str | int],
            )

            response_dict = json.loads(response)

            sota_submit_idx = response_dict.get("selected_SOTA_idx", None)

            if sota_submit_idx and int(sota_submit_idx) - 1 < len(sota_exp_fb_list):
                sota_submit = sota_exp_fb_list[int(sota_submit_idx) - 1]
                sota_idx_in_trace = trace.hist.index(sota_submit)
                logger.info(
                    f"Auto SOTA selector: selected SOTA experiment No. {sota_submit_idx} to submit, which is the No. {sota_idx_in_trace + 1} in the trace"
                )
                return sota_submit[0]
            else:
                # no SOTA experiment to submit, using the latest SOTA experiment
                if len(sota_exp_fb_list) > 0:
                    logger.info("Auto SOTA selector: No SOTA experiment to submit, using the latest SOTA experiment")
                    return sota_exp_fb_list[-1][0]
                else:
                    logger.info("Auto SOTA selector: No SOTA experiment in trace yet")
                    return None


class BestValidSelector(SOTAexpSelector):
    def get_sota_exp_to_submit(
        self, trace: Trace, num: int = 1, use_decision: bool = True, each_trace: bool = False
    ) -> DSExperiment | None:
        direction_sign = 1 if trace.scen.metric_direction else -1

        def get_sort_key(exp_fb: tuple[DSExperiment, ExperimentFeedback]) -> tuple[bool, float]:
            score = -np.inf
            # default score is alway the smallest
            result: pd.DataFrame | None = exp_fb[0].result
            if result is not None:
                score = direction_sign * result.loc["ensemble"].iloc[0]
            if use_decision:
                return score
            else:
                return (exp_fb[1].decision, score)

        if not each_trace:
            sota_exp_fb_list = trace.experiment_and_feedback_list_after_init(return_type="all", search_type="all")
        else:
            sota_exp_fb_list = []

        if not sota_exp_fb_list:
            for i in trace.get_leaves():
                fb_list = trace.experiment_and_feedback_list_after_init(
                    return_type="all",
                    search_type="ancestors",
                    selection=(i,),
                )
                if fb_list:
                    fb_list = sorted(fb_list, key=get_sort_key, reverse=True)
                    sota_exp_fb_list.extend(fb_list[: max(num // len(trace.get_leaves()), 1)])

        if len(sota_exp_fb_list) == 0:
            logger.info("Best Valid SOTA selector: No SOTA in trace yet")
            return None
        else:
            sota_exp_fb_list = sorted(sota_exp_fb_list, key=get_sort_key, reverse=True)
            if not each_trace:
                print([get_sort_key(i) for i in sota_exp_fb_list])
                return [i[0] for i in sota_exp_fb_list[:num]]
            else:
                for i in list(set(sota_exp_fb_list)):
                    if not i[1].decision:
                        selected_sota_exp = [i[0] for i in list(set(sota_exp_fb_list))[:num]]
                        selected_index = trace.exp2idx(selected_sota_exp)
                return [i[0] for i in list(set(sota_exp_fb_list))[:num]]


def process_experiment(i, competition, folder, grade_py_code, subfolder_name, trace):
    """
    Worker function to process a single experiment in an isolated directory.
    This function is designed to be called by a multiprocessing pool.
    """
    # 1. Get a unique identifier for the experiment
    loop_id = trace.idx2loop_id[trace.exp2idx(i)]

    # --- Result variables to be returned ---
    execute_ret_code = 1
    grade_stdout = ""
    input_folder = T("scenarios.data_science.share:scen.input_path").r()

    try:
        # Set up the isolated environment
        implementation = FBWorkspace()
        implementation.inject_code_from_file_dict(i.experiment_workspace)
        mock_folder = f"/tmp/mock/{competition}/{input_folder}"

        # Run the script with its CWD set to the temporary folder
        env = get_ds_env(
            extra_volumes={mock_folder: input_folder},
            running_timeout_period=DS_RD_SETTING.full_timeout,
        )
        result = implementation.run(env=env, entry="python main.py")
        stdout = re.sub(r"^chmod:.*\n?", "", result.stdout, flags=re.MULTILINE)
        execute_ret_code = result.exit_code
        logger.info(f"{competition}/{loop_id}/main.py, execute_ret_code: {execute_ret_code}, stdout: {stdout}")

        # Run the grading script if the model script succeeded
        if execute_ret_code == 0:
            env = get_ds_env(extra_volumes={mock_folder: input_folder})
            implementation.inject_files(**{"grade.py": grade_py_code})
            result = implementation.run(env=env, entry="python grade.py")
            grade_stdout = re.sub(r"^chmod:.*\n?", "", result.stdout, flags=re.MULTILINE)
            execute_ret_code = result.exit_code
            logger.info(
                f"grade for {competition}/{loop_id}/main.py, execute_ret_code: {execute_ret_code}, stdout: {stdout}"
            )
        else:
            logger.info(f"Skipping grading for {competition}/{loop_id}/main.py due to execution failure.")

    except Exception as e:
        logger.info(f"CRITICAL ERROR while processing experiment {competition}/{loop_id}/main.py: {e}")

    score = None
    if grade_stdout:
        try:
            score = float(json.loads(grade_stdout)["score"])
        except:
            try:
                score = float(eval(grade_stdout)["score"])
            except:
                try:
                    score = float(re.findall(r"[-+]?\d*\.\d+|\d+", grade_stdout)[-1])
                except:
                    logger.info(f"Failed to extract score from grade_stdout: {grade_stdout}")

    return loop_id, score


def check_hit(selected_sota_exp: list[int], trace: Trace, sota_result: dict[str, Any]):
    selected_index = trace.exp2idx(selected_sota_exp)
    hit = False
    for i in selected_index:
        if hasattr(trace, "idx2loop_id") and i in trace.idx2loop_id:
            selected_loop = trace.idx2loop_id[i]
            if selected_loop in sota_result["medal_loops"]:
                hit = True
                break
        else:
            if i in sota_result["medal_loops_index"]:
                hit = True
                break
    return hit


def select_one_trace(selector_name, trace_pkl_path, trace_folder, num=1, use_decision=True, each_trace=False):
    input_folder = T("scenarios.data_science.share:scen.input_path").r()
    competition = trace_pkl_path.stem.split(".")[0]
    mock_folder = f"/tmp/mock/{competition}"

    sota_result = json.load(open(trace_folder / f"{trace_pkl_path.stem.split('_')[0]}_loops.json", "r"))
    if not sota_result["medal_loops"]:
        logger.info(f"Selector {selector_name} found no SOTA loops in trace: {trace_pkl_path}, skipping...")
        return "", False

    if selector_name == "global":
        selector = GlobalSOTASelector()
    elif selector_name == "auto":
        selector = AutoSOTAexpSelector()
    elif selector_name == "best_valid":
        selector = BestValidSelector()

    trace = pickle.load(trace_pkl_path.open("rb"))
    subfolder_name = trace_pkl_path.parent.name
    # fix metric direction for detecting-insults-in-social-commentary   
    if competition == "detecting-insults-in-social-commentary":
        trace.scen.metric_direction = 1

    selected_sota_exp = BestValidSelector().get_sota_exp_to_submit(trace, 1, True, False)
    hit = check_hit(selected_sota_exp, trace, sota_result)
    if hit:
        logger.info(f"Best selector Hit {hit} SOTA experiment for {competition}")
        return competition, hit

    selected_sota_exp = selector.get_sota_exp_to_submit(trace, num, use_decision, each_trace)
    hit = check_hit(selected_sota_exp, trace, sota_result)

    if not hit:
        logger.info(f"Trace selector not Hit SOTA experiment for {competition}")
        return competition, hit

    (Path(mock_folder) / input_folder).mkdir(parents=True, exist_ok=True)

    system_prompt = T(".prompts:sample_data.system").r()
    implementation = FBWorkspace()
    implementation.inject_code_from_file_dict(selected_sota_exp[0].experiment_workspace)
    reference_code = implementation.file_dict.get("main.py")
    grade_py_code = ""
    data_py_path = Path(mock_folder) / "data.py"
    grade_py_path = Path(mock_folder) / "grade.py"
    if not (Path(mock_folder) / input_folder / "label.csv").exists():
        logger.info(f"Generating {data_py_path}...")
        err_msg = ""
        for retry in range(5):
            user_prompt = T(".prompts:sample_data.user").r(
                reference_code=reference_code, error=err_msg, input_folder=input_folder
            )
            data_py_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )

            if not "label.csv" in data_py_code:
                err_msg = f"Please make sure `./{input_folder}/label.csv` will be generated. "
                continue

            logger.info("[data.py] Running...")
            env = get_ds_env(
                extra_volumes={
                    str((Path(mock_folder) / input_folder)): input_folder,
                    f"{DS_RD_SETTING.local_data_path}/{competition}": "./source",
                },
                running_timeout_period=DS_RD_SETTING.full_timeout,
            )
            implementation.inject_files(**{"data.py": data_py_code})
            result = implementation.run(env=env, entry="python data.py")
            stdout = re.sub(r"^chmod:.*\n?", "", result.stdout, flags=re.MULTILINE)
            execute_ret_code = result.exit_code
            logger.info(f"[data.py] execute_ret_code: {execute_ret_code}")
            logger.info(stdout)
            if execute_ret_code == 0:
                # write data code to data_py_path
                data_py_path.write_text(data_py_code)
                result = implementation.run(env=env, entry="python main.py")
                stdout = re.sub(r"^chmod:.*\n?", "", result.stdout, flags=re.MULTILINE)
                execute_ret_code = result.exit_code
                logger.info(f"[main.py] execute_ret_code: {execute_ret_code}")
                logger.info(stdout)
                if execute_ret_code == 0:
                    if not grade_py_path.exists():
                        for retry in range(5):
                            user_prompt = T(".prompts:grade.user").r(
                                reference_code=reference_code,
                                sample_code=data_py_code,
                                input_folder=input_folder,
                                error=err_msg,
                            )
                            grade_py_code = PythonAgentOut.extract_output(
                                APIBackend().build_messages_and_create_chat_completion(
                                    user_prompt=user_prompt,
                                    system_prompt=system_prompt,
                                )
                            )
                            implementation.inject_files(**{"grade.py": grade_py_code})
                            result = implementation.run(env=env, entry="python grade.py")
                            stdout = re.sub(r"^chmod:.*\n?", "", result.stdout, flags=re.MULTILINE)
                            execute_ret_code = result.exit_code
                            logger.info(f"[grade.py] execute_ret_code: {execute_ret_code}")
                            logger.info(stdout)
                            if execute_ret_code == 0:
                                grade_py_path.write_text(grade_py_code)
                                break
                            else:
                                err_msg = f"Error in grade.py: {shrink_text(stdout, context_lines=20, line_len=500)}"
                    break
                else:
                    err_msg = f"Error in main.py: {shrink_text(stdout, context_lines=20, line_len=500)}"
            else:
                err_msg = f"Error in data.py: {shrink_text(stdout, context_lines=20, line_len=500)}"
    else:
        if not grade_py_path.exists():
            data_py_code = Path(data_py_path).read_text()
            user_prompt = T(".prompts:grade.user").r(
                reference_code=reference_code, sample_code=data_py_code, input_folder=input_folder, error=""
            )
            grade_py_code = PythonAgentOut.extract_output(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            )
            grade_py_path.write_text(grade_py_code)
        grade_py_code = grade_py_path.read_text()

    results = multiprocessing_wrapper(
        [
            (process_experiment, (i, competition, mock_folder, grade_py_code, subfolder_name, trace))
            for i in selected_sota_exp
        ],
        n=5,
    )
    if results:
        logger.debug(f"{results =}")
        results = [(i, j) for i, j in results if j is not None]
        direction_sign = 1 if trace.scen.metric_direction else -1
        results.sort(key=lambda x: x[1] * direction_sign, reverse=True)
        if results:
            selected_loop = results[0][0]
            hit = selected_loop in sota_result["medal_loops"]

    return competition, hit


# TODO: more advanced sota exp selector (e.g. LLM-based, merge exp with multiple sub-trace)
def select_on_existing_trace(
    selector_name: str,
    trace_root,
):
    """
    Offline select SOTA experiment from existing trace.
    :param selector_name: name of the selector to use
    :param trace_folder: folder containing the trace
    """
    result_dict = {}
    for trace_folder in Path(trace_root).iterdir():
        if not trace_folder.is_dir():
            continue
        if not "devoted-burro" in str(trace_folder):
            # if not "devoted-burro" in str(trace_folder):
            continue
        trace_folder = Path(trace_folder)

        num = 5
        use_decision = True
        each_trace = True

        hit_list = multiprocessing_wrapper(
            [
                (select_one_trace, (selector_name, trace_pkl_path, trace_folder, num, use_decision, each_trace))
                for trace_pkl_path in trace_folder.glob("*.pkl")
            ],
            n=1,
        )
        hit_count = sum([i[1] for i in hit_list])
        print(
            f"Selector {selector_name} {num} {use_decision} {each_trace} hit {hit_count} out of {len(hit_list)} traces, hit rate: {hit_count / len(hit_list) * 100:.2f}%"
        )
        result_dict[trace_folder.name] = {
            "hit": hit_count,
            "total": len(hit_list),
            "hit_rate": hit_count / len(hit_list) * 100,
            "hit_list": hit_list,
        }
    all_hit = sum([result["hit"] for result in result_dict.values()])
    all_total = sum([result["total"] for result in result_dict.values()])
    result_dict["all"] = {
        "hit": all_hit,
        "total": all_total,
        "hit_rate": all_hit / all_total * 100 if all_total > 0 else 0,
    }
    json.dump(result_dict, open(f"result_{selector_name}.json", "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(select_on_existing_trace)
