import json
from pathlib import Path
import pickle
import random
from typing import Dict, Tuple

import fire
import numpy as np
import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExperimentFeedback, SOTAexpSelector, Trace
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.utils.agent.tpl import T
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
    def get_sota_exp_to_submit(self, trace: Trace) -> DSExperiment | None:
        sota_exp_fb_list = trace.experiment_and_feedback_list_after_init(return_type="all", search_type="all")
        direction_sign = 1 if trace.scen.metric_direction else -1

        def get_sort_key(exp_fb: tuple[DSExperiment, ExperimentFeedback]) -> tuple[bool, float]:
            score = -np.inf
            result: pd.DataFrame | None = exp_fb[0].result
            if result is not None:
                score = result.loc["ensemble"].iloc[0]
            return (exp_fb[1].decision, direction_sign * score)

        if len(sota_exp_fb_list) == 0:
            logger.info("Best Valid SOTA selector: No SOTA in trace yet")
            return None
        else:
            sota_exp_fb_list = sorted(sota_exp_fb_list, key=get_sort_key, reverse=True)
            return sota_exp_fb_list[0][0]


def select_one_trace(selector_name, trace_pkl_path, trace_folder):
    sota_result = json.load(open(trace_folder / f"{trace_pkl_path.stem.split('_')[0]}_loops.json", "r"))
    if not sota_result['medal_loops']:
        logger.info(
            f"Selector {selector_name} found no SOTA loops in trace: {trace_pkl_path}, skipping..."
        )
        return False

    if selector_name == "global":
        selector = GlobalSOTASelector()
    elif selector_name == "auto":
        selector = AutoSOTAexpSelector()
    elif selector_name == "best_valid":
        selector = BestValidSelector()

    trace = pickle.load(trace_pkl_path.open("rb"))
    selected_sota_exp = selector.get_sota_exp_to_submit(trace)

    selected_index = trace.exp2idx(selected_sota_exp)
    if hasattr(trace, "idx2loop_id") and selected_index in trace.idx2loop_id:
        selected_loop = trace.idx2loop_id[selected_index]
        logger.info(
            f"Selector {selector_name} selected SOTA experiment: { sota_result['medal_loops']}, loop in trace: {selected_loop}, trace: {trace_pkl_path}"
        )
        return selected_loop in sota_result["medal_loops"]
    else:
        logger.info(
            f"Selector {selector_name} selected SOTA experiment: { sota_result['medal_loops_index']}, index in trace: {selected_index}, trace: {trace_pkl_path}"
        )
        return selected_index in sota_result["medal_loops_index"]


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
        trace_folder = Path(trace_folder)

        # hit_list = []
        # for trace_pkl_path in trace_folder.glob("*.pkl"):
        #     hit_list.append(select_one_trace(selector_name, trace_pkl_path, trace_folder))

        hit_list = multiprocessing_wrapper(
            [
                (select_one_trace, (selector_name, trace_pkl_path, trace_folder))
                for trace_pkl_path in trace_folder.glob("*.pkl")
            ],
            n=8,
        )

        print(
            f"Selector {selector_name} hit {sum(hit_list)} out of {len(hit_list)} traces, hit rate: {sum(hit_list) / len(hit_list) * 100:.2f}%"
        )
        result_dict[trace_folder.name] = {
            "hit": sum(hit_list),
            "total": len(hit_list),
            "hit_rate": sum(hit_list) / len(hit_list) * 100,
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
