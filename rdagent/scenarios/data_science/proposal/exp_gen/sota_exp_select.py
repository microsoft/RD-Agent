import json
import random
from typing import Dict, Tuple

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.proposal import SOTAexpSelector, Trace
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

        sota_exp_fb_list = trace.experiment_and_feedback_list_after_init(return_type="sota", search_type="all")

        if len(sota_exp_fb_list) == 0:
            logger.info("Auto SOTA selector: No SOTA in trace yet")
            return None

        elif len(sota_exp_fb_list) == 1:
            logger.info("Auto SOTA selector: Only one SOTA in trace, using it")
            return sota_exp_fb_list[0][0]

        else:
            logger.info("Auto SOTA selector: Multiple SOTA in trace, calling LLM to select the best one")

            SOAT_exp_with_desc_and_scores = "Historical SOTA experiments:\n\n"

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

            if sota_submit_idx is not None:
                sota_submit = sota_exp_fb_list[int(sota_submit_idx) - 1]
                sota_idx_in_trace = trace.hist.index(sota_submit)
                logger.info(
                    f"Auto SOTA selector: selected SOTA experiment No. {sota_submit_idx} to submit, which is the No. {sota_idx_in_trace + 1} in the trace"
                )
                return sota_submit[0]
            else:
                # no SOTA experiment to submit, using the latest SOTA experiment
                logger.info("Auto SOTA selector: No SOTA experiment to submit, using the latest SOTA experiment")
                return sota_exp_fb_list[-1][0]


# TODO: more advanced sota exp selector (e.g. LLM-based, merge exp with multiple sub-trace)
