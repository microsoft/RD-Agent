import json
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import ExperimentFeedback, SOTAexpSelector, Trace
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

                sota_exp_fb_list = new_sota_exp_fb_list

                if len(sota_exp_fb_list) == 0:
                    logger.info("Auto SOTA selector: No SOTA in trace yet")
                    return None

                elif len(sota_exp_fb_list) == 1:
                    logger.info("Auto SOTA selector: Only one SOTA in trace, using it")
                    return sota_exp_fb_list[0][0]
                else:
                    logger.info(
                        f"Auto SOTA selector: {len(sota_exp_fb_list)} SOTA experiments found in all traces, calling LLM to select the best one"
                    )

            for i, (exp, ef) in enumerate(sota_exp_fb_list):
                if exp:
                    current_final_score = pd.DataFrame(exp.result).loc["ensemble"].iloc[0]
                    desc = T("scenarios.data_science.share:describe.exp").r(
                        exp=exp, heading="SOTA of previous exploration of the scenario"
                    )
                    SOAT_exp_with_desc_and_scores += f"""SOTA experiment No. {i+1}:
                        Description: {desc}
                        Final score: {current_final_score}\n\n"""

            system_prompt = T(".prompts_selector:auto_sota_selector.system").r(
                scenario=trace.scen.get_scenario_all_desc()
            )

            user_prompt = T(".prompts_selector:auto_sota_selector.user").r(
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


# TODO: more advanced sota exp selector (e.g. LLM-based, merge exp with multiple sub-trace)
