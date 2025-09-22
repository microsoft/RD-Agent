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
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction

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


class AutoEMAelector(SOTAexpSelector):

    def __init__(self, num_candidates: int = 1, use_decision: bool = True, each_trace: bool = False):
        """
        Args:
            num_candidates (int): The number of top experiments to return.
            use_decision (bool): If True, filters out experiments marked with a negative decision.
            each_trace (bool): If True, selects top candidates from each branch instead of globally.
        """
        self.num_candidates = num_candidates
        self.use_decision = use_decision
        self.score_window_size = 5  # EMA window size
        self.str_window_size = 5  # EMA window size for 

    def _get_path(self, node, parent_nodes):
        # FIXME: we should remove it in the future.
        path = [node]
        parent = parent_nodes.get(node)
        if parent is not None:
            path.extend(self._get_path(parent, parent_nodes))
        return path

    def get_sota_exp_to_submit(self, trace: Trace, **kwargs) -> DSExperiment | None:
        """
        Sorts all valid experiments by score and returns the top N.
        """
        top_experiments = self.collect_sota_candidates(trace)
        if top_experiments:
            return top_experiments[0]
        return None
    
    def _get_node_parents(self, trace: Trace) -> Tuple[Dict[int, Optional[int]], Dict[int, Optional[int]]]:
        root_nodes = {}
        parent_nodes = {}
        for node in range(len(trace.hist)):
            parents = trace.get_parents(node)
            root_nodes[node] = parents[0]
            parent_nodes[node] = parents[-2] if len(parents) > 1 else None
        if hasattr(trace, "idx2loop_id"):
            root_nodes = {trace.idx2loop_id[n]: trace.idx2loop_id[r] for n, r in root_nodes.items()}
            parent_nodes = {
                trace.idx2loop_id[n]: trace.idx2loop_id[r] if r is not None else r
                for n, r in parent_nodes.items()
            }
        return root_nodes, parent_nodes
    

    def llm_ema(self,sub_trace_df:pd.DataFrame, scenario_desc,window_size: int) -> List[float]:
        n = len(sub_trace_df)
        #grouped_exp_lists = []
        sub_trace_df['LLM_Overall_Rating'] = 0.0
        for start_idx in range(0, n, window_size):
            end_idx = min(start_idx + window_size, n)
            window_df = sub_trace_df.iloc[start_idx:end_idx]

            exp_list = []
            for _, row in window_df.iterrows():
                exp_str = (
                    f"Hypothesis: {row['Hypothesis']}\n"
                    f"Score: {row['Score']}\n"
                    f"Code: {row['Code']}\n"
                    f"Stdout: {row['Stdout']}\n"
                )
                exp_list.append(exp_str)

            sys_prompt = T(".prompts:auto_sota_selector_ema.system").r(

        )
    
            user_prompt = T(".prompts:auto_sota_selector_ema.user").r(
                exp_list=exp_list,
                window_size=window_size,
                num_experiments=len(exp_list),
                scenario_desc=scenario_desc
            )
            response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt)

            task_dict = json.loads(response)
            for i, item in enumerate(task_dict):
                original_idx = window_df.index[i]  # window_df 的真实 DataFrame 索引
                sub_trace_df.at[original_idx, 'Overall_Rating'] = float(item['Overall_Rating'])
                sub_trace_df.at[original_idx, 'Rating_Reason'] = item['Reason']

        return sub_trace_df['LLM_Overall_Rating'].tolist()  
    

    def ema(self,series, window):
        alpha = 2 / (window + 1)
        ema_values = []
        ema_prev = series.iloc[0]
        for val in series:
            ema_prev = alpha * val + (1 - alpha) * ema_prev
            ema_values.append(ema_prev)
        return ema_values

    def collect_sota_candidates(self, trace: Trace) -> list[DSExperiment] | None:

        competition = trace.scen.competition
        bigger_is_better = get_metric_direction(competition)
        loop_number  = len(trace.hist)
        trace_path_number = len(set(root_nodes.values()))
        root_nodes, parent_nodes = self._get_node_parents(trace=trace)
        loop_id2idx = {v: k for k, v in trace.idx2loop_id.items()}
        summary_list = pd.DataFrame(list(root_nodes.items()), columns=['Node', 'Root'])

        scenario_desc = trace.scen.get_scenario_all_desc(eda_output=None)
        
        ID_LIST = []
        for i in range(trace_path_number):
            sub_trace_summary = []
            sub_trace_list = summary_list[summary_list['Root'] == i]['Node'].tolist()
            for sub_id in sub_trace_list:
                hypo_str = trace.hist[loop_id2idx[sub_id]][0].hypothesis.hypothesis
                score = trace.hist[loop_id2idx[sub_id]][0].result.loc["ensemble"].iloc[0].round(5)
                code_str = trace.hist[loop_id2idx[sub_id]][0].experiment_workspace.file_dict['main.py']
                time_cost = trace.hist[loop_id2idx[sub_id]][0].experiment_workspace.running_info.running_time
                std_out =  trace.hist[loop_id2idx[sub_id]][0].experiment_workspace.file_dict['stdout.txt']
                sub_trace_summary.append((sub_id, hypo_str, score, code_str, time_cost, std_out))
            sub_trace_df = pd.DataFrame(sub_trace_summary, columns=['loop id','Hypothesis', 'Score', 'Code', 'Time', 'Stdout'])

            sub_trace_df['EMA_Score'] = self.ema(sub_trace_df['Score'], self.score_window_size)
            sub_trace_df['Overfit_Risk'] = sub_trace_df['Score_Deviation'].abs() / (sub_trace_df['EMA_Score'].replace(0, np.nan))
            sub_trace_df['Overfit_Risk'] = sub_trace_df['Overfit_Risk'].fillna(-100000)
            sub_trace_df["Mean_Score"] = sub_trace_df['Score'].mean()
            llm_ema_scores = self.llm_ema(sub_trace_df, scenario_desc, self.str_window_size)
            sub_trace_df['LLM_Overall_Rating'] = llm_ema_scores

            llm_scores = sub_trace_df['LLM_Overall_Rating']
            risk_scores = sub_trace_df['Overfit_Risk']
            risk_scores_scaled = 1 - (risk_scores / risk_scores.max())
            sub_trace_df['Combined_Score'] = llm_scores * 0.7 + risk_scores_scaled * 0.3  # 权重可调
            best_exp_id = sub_trace_df.loc[sub_trace_df['Combined_Score'].idxmax()]['loop id']
            best_score = sub_trace_df.loc[best_exp_id]['Score']
            best_code = sub_trace_df.loc[best_exp_id]['Code']
            if best_exp_id in ID_LIST:
                ID_LIST.append(best_exp_id,best_score,best_code)

        return ID_LIST




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