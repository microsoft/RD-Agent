import pandas as pd
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiEvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolvable_subjects import FBWorkspace
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    CoSTEERQueriedKnowledge,
    MultiProcessEvolvingStrategy
)
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator
from rdagent.core.exception import RunnerError
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.dev.runner_mcts.eval import DSRunnerMCTSEvaluator,DSRunnerFeedback
from rdagent.utils.agent.ret import PythonBatchEditOut, PythonBatchPatchOut
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry

from typing import Generic, Optional, NamedTuple, Callable, Union
from typing import Generic, TypeVar, Protocol
from abc import ABC, abstractmethod
import itertools
import torch
import math
import json
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.core.evolving_framework import EvoStep
from rdagent.scenarios.kaggle.kaggle_crawler import get_metric_direction
import time
class DSRunnerMCSTCoSTEERSettings(CoSTEERSettings):
    """Data Science CoSTEER settings"""

    class Config:
        env_prefix = "DS_Runner_MCTS_CoSTEER_"

    max_seconds_multiplier: int = 1
    env_type: str = "docker"
    diff_mode: bool = False
    # TODO: extract a function for env and conf.



class MCTSNode:
    def __init__(self, workspace, parent=None):
        self.workspace = workspace
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.untried_actions : list[dict] = []  #
        self.score = 0
        self.feedback: DSRunnerFeedback = None
    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0


class DSRunnerMCTSMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, scen, settings,max_iterations=2, exploration_c=1.4):
        super().__init__(scen=scen,settings=settings)
        self.max_iterations = max_iterations
        self.exploration_c = exploration_c
        self.root = MCTSNode(workspace=None)
        self.KEY_CHANGE_SUMMARY = "__change_summary__"

    def evolve(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        evolving_trace: list[EvoStep] = [],
        **kwargs,
    ) -> EvolvingItem:
        code_list = [None for _ in range(len(evo.sub_tasks))]

        # 1.找出需要evolve的task
        to_be_finished_task_index: list[int] = []
        for index, target_task in enumerate(evo.sub_tasks):
            target_task_desc = target_task.get_task_information()
            if target_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                # NOTE: very weird logic:
                # it depends on the knowledge to set the already finished task
                code_list[index] = queried_knowledge.success_task_to_knowledge_dict[
                    target_task_desc
                ].implementation.file_dict
            elif (
                target_task_desc not in queried_knowledge.success_task_to_knowledge_dict
                and target_task_desc not in queried_knowledge.failed_task_info_set
            ):
                to_be_finished_task_index.append(index)

        last_feedback = None
        if len(evolving_trace) > 0:
            last_feedback = evolving_trace[-1].feedback
            assert isinstance(last_feedback, CoSTEERMultiFeedback)

        mcts_node_list,code_dict,feedback  = self.implement_one_task(
                evo.sub_tasks[to_be_finished_task_index[0]],
                queried_knowledge,
                evo.experiment_workspace,
            )
        code_list[to_be_finished_task_index[0]] = code_dict
        
        evo = self.assign_code_list_to_evo(code_list, evo)
        evo.MCTS_NODE_LIST = mcts_node_list
        evo.FEEDBACK = feedback
        return evo

    @wait_retry(retry_n=5)       
    def generate_modifications(
        self,
        target_task: CoSTEERTask,
        workspace: FBWorkspace,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        num_candidates: int = 2,  
    ) -> list[dict[str, str]]:
        """
        Generate multiple candidate modifications for a task using LLM.
        Returns a list of dicts [{filename: new_content}], each dict is a candidate modification.
        """

        task_info = target_task.get_task_information()
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[task_info] if queried_knowledge else []
        )
        queried_former_failed_knowledge = queried_former_failed_knowledge[0] if queried_former_failed_knowledge else None


        system_prompt = T(".prompts:DSCoSTEER.system_refine").r(
                out_spec=PythonBatchEditOut.get_spec(with_del=False),
                diff_mode=self.settings.diff_mode,
                num_candidates = num_candidates
            )

        session = APIBackend().build_chat_session(session_system_prompt=system_prompt)

        user_prompt = T(".prompts:DSCoSTEER.user").r(
            code=workspace.all_codes,
            change_summary=workspace.change_summary,
            queried_former_failed_knowledge=queried_former_failed_knowledge,
            num_candidates=num_candidates,
        )

        code_raw = session.build_chat_completion(user_prompt=user_prompt)

        try:
            code_candidates = json.loads(code_raw)
        except Exception:
            if self.settings.diff_mode:
                single_candidate = PythonBatchPatchOut.extract_output(code_raw, prefix=workspace.workspace_path)
            else:
                single_candidate = PythonBatchEditOut.extract_output(code_raw)
            code_candidates = [single_candidate]

        final_candidates = []
        for candidate in code_candidates:
            candidate_filtered = {k: v for k, v in candidate.items() if k in workspace.file_dict.keys()}
            final_candidates.append(candidate_filtered)

        logger.info(f"Generated {len(final_candidates)} candidate modifications.")
        return final_candidates

    def reward_function(self,fb:DSRunnerFeedback) ->float:
        score = getattr(fb, "score", 0.0)
        code_score = 1.0 if getattr(fb, "code_accept", False) else 0
        acceptable_score = 1.0 if getattr(fb, "acceptable", False) else 0
        bigger_is_better = get_metric_direction(self.scen.competition)
        if score:
            normalized_score = 1 / (1 + math.exp(- score)) # sigmoid smooth
            if not bigger_is_better:
                normalized_score = 1.0 - normalized_score
            reward  = normalized_score#code_score + acceptable_score + normalized_score
        else:
            reward = -1
        return reward
    

    def select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            ucb_values = [
                (child.value + self.exploration_c * math.sqrt(math.log(node.visit_count + 1) / (child.visit_count + 1)))
                for child in node.children
            ]
            max_ucb = max(ucb_values)
            best_children = [child for child, ucb in zip(node.children, ucb_values) if ucb == max_ucb]
            node = best_children[0]  # Break ties arbitrarily
        return node

    def expand(
        self,
        node: MCTSNode,
        target_task: CoSTEERTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
    ) -> Optional[MCTSNode]:
        """
        Expand a node in MCTS:
        - Generate multiple candidate modifications if this node has not been visited.
        - Pop one untried action to create a child node.
        - Return the newly created child node.
        """

        if node.visit_count == 0 and (not node.untried_actions):
            modifications_list = self.generate_modifications(
                target_task, node.workspace, queried_knowledge, num_candidates=DS_RD_SETTING.mcts_hypothesis_sample_size
            )
            node.untried_actions = modifications_list
            if node.children is None:
                node.children = []

        if not node.untried_actions:
            return None  

        action = node.untried_actions.pop()
        new_workspace = FBWorkspace(target_task=target_task)
        new_workspace.inject_files(**node.workspace.file_dict)
        new_workspace.inject_files(**action)
        child_node = MCTSNode(workspace=new_workspace, parent=node)
        node.children.append(child_node)
        return child_node

    def expand_batch(self, node: MCTSNode, target_task, queried_knowledge=None, batch_size=DS_RD_SETTING.mcts_hypothesis_sample_size)-> list[MCTSNode]:
        if not node.untried_actions:
            node.untried_actions = self.generate_modifications(
                target_task, node.workspace, queried_knowledge, num_candidates=DS_RD_SETTING.mcts_hypothesis_sample_size
            )
            if node.children is None:
                node.children = []

        new_nodes = []
        for _ in range(min(batch_size, len(node.untried_actions))):
            action = node.untried_actions.pop(0)
            new_workspace = FBWorkspace(target_task=target_task)
            new_workspace.inject_files(**node.workspace.file_dict)
            new_workspace.inject_files(**action)
            child_node = MCTSNode(workspace=new_workspace, parent=node)
            node.children.append(child_node)
            new_nodes.append(child_node)
        return new_nodes
    
    def simulate(self, node: MCTSNode, target_task, gt_workspace, queried_knowledge):
        evaluator = DSRunnerMCTSEvaluator(scen=self.scen)
        logger.info("Simulating node with workspace files")
        feedback = evaluator.evaluate(target_task, node.workspace, gt_workspace, queried_knowledge)
        reward = feedback.score if feedback.score is not None else (-1 if not feedback.acceptable else 0)
        return reward

    def simulate_batch(self, nodes: list[MCTSNode], target_task, gt_workspace, queried_knowledge, n_processes, estimated_time_sec) -> list[float]:
        """
        Parallel simulation of multiple MCTS nodes using multiprocessing_wrapper.
        Returns the reward list for each node.
        """
        evaluator = DSRunnerMCTSEvaluator(scen=self.scen)
        func_calls = [
            (evaluator.evaluate, (target_task, node.workspace, gt_workspace, queried_knowledge,estimated_time_sec,False))
            for node in nodes
        ]

        feedbacks = multiprocessing_wrapper(func_calls, n=n_processes)
        # rewards = [
        #     fb.score if fb.score is not None else (-1 if not fb.acceptable else 0)
        #     for fb in feedbacks
        # ]
        rewards = [self.reward_function(fb) for fb in feedbacks]
        for node,fb in zip(nodes,feedbacks):
            node.score = fb.score
            node.feedback = fb
        return rewards

    def backpropagate_batch(self, nodes: list[MCTSNode], rewards: list[float]):
        """
        Backpropagate a batch of rewards for a batch of nodes.
        """
        for node, reward in zip(nodes, rewards):
            current = node
            while current is not None:
                current.visit_count += 1
                current.value_sum += reward
                current = current.parent

    def backpropagate(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent

    @wait_retry(retry_n=5)
    def implement_one_task(
        self,
        target_task: CoSTEERTask,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
    ) -> dict[str, str]:

        MCTS_NODE_LIST = []
        root = MCTSNode(workspace=workspace)
        logger.info("Starting root node !")
        evaluator = DSRunnerMCTSEvaluator(scen=self.scen)
        begin_time = time.time()
        feedback_root = evaluator.evaluate(target_task, root.workspace, workspace, queried_knowledge,time_max= 3600,root= True)
        root.feedback= feedback_root
        end_time = time.time()
        elapsed_time1 = end_time - begin_time
        logger.info(elapsed_time1)
        elapsed_time = feedback_root.cost_time*1.5
        root.score = feedback_root.score
        MCTS_NODE_LIST.append(root)
        
        root_is_none = (feedback_root.score is None) and (feedback_root.code_accept == False)
        bigger_is_better = get_metric_direction(self.scen.competition)
        stdout = feedback_root.stdout

        system_prompt = T(".prompts:DSCoSTEER_mcts.system").r(
            scenario=self.scen.get_scenario_all_desc(eda_output=root.workspace.file_dict.get("EDA.md", None)),
            task_desc=target_task.get_task_information(),
        )
        user_prompt = T(".prompts:DSCoSTEER_mcts.user").r(
            code=root.workspace.all_codes,
            stdout = stdout,
            elapsed_time = elapsed_time,
            bigger_is_better = bigger_is_better
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format= {"type": "json_object"},
        )
        response = json.loads(response)  # 将字符串解析为 JSON 对象

        enter_mcts = response["enter_mcts"]
        estimated_time_sec = response["estimated_time_sec"]
        gpu_count = response["gpu_count"]
        recommended_search_depth = response["recommended_search_depth"]
        #reasoning_text = response["reasoning"]["text"]
        confidence = response["reasoning"]["confidence"]
        
        enter_condition = (DS_RD_SETTING.runner_max_loop>1 and DS_RD_SETTING.enable_runner_mcts and not root_is_none and confidence> 75 and enter_mcts)

        if enter_condition:
            if DS_RD_SETTING.multiprocessing_mcts_simulation is not True:
                logger.info(f"Starting MCTS with max_iterations={self.max_iterations}")
                search_depth = min(recommended_search_depth,self.max_iterations)
                count_processes = min(gpu_count,DS_RD_SETTING.mcts_n_processes)
                for _ in range(self.max_iterations):
                    logger.info(f"Starting iteration {_+1}/{self.max_iterations}")
                    node = self.select(root)
                    new_node = self.expand(node, target_task, queried_knowledge)
                    logger.info(f"Expanded node. Current root visit count: {root.visit_count}")
                    logger.info(f"node children: {node.children}")
                    reward = self.simulate(new_node, target_task, workspace, queried_knowledge)
                    logger.info(f"Simulation reward: {reward}")
                    self.backpropagate(new_node, reward)
                    logger.info(f"Iteration completed. Current root visit count: {root.visit_count}")
                    MCTS_NODE_LIST.append(new_node)
                #best_child = max(root.children, key=lambda c: c.value)
            else:
                search_depth = min(recommended_search_depth,self.max_iterations)
                count_processes = min(gpu_count,DS_RD_SETTING.mcts_n_processes)
                for _ in range(search_depth):
                    logger.info(f"Starting Mutiprocessing iteration !")
                    logger.info(f"Starting MCTS with max_iterations={_}")
                    node = self.select(root)
                    logger.info(f"Selected node with visit count {node.visit_count} and value {node.value}")
                    new_nodes = self.expand_batch(node, target_task, queried_knowledge, batch_size=DS_RD_SETTING.mcts_multiprocessing_batch_size)
                    rewards = self.simulate_batch(new_nodes, target_task, workspace, queried_knowledge, n_processes=count_processes,estimated_time_sec=estimated_time_sec)
                    self.backpropagate_batch(new_nodes, rewards)
                    MCTS_NODE_LIST.append(new_nodes)
                #best_child = max(root.children, key=lambda c: c.value)
            all_nodes = []
            for item in MCTS_NODE_LIST:
                if isinstance(item, list):
                    all_nodes.extend(item) 
                else:
                    all_nodes.append(item)
            if bigger_is_better:
                best_child = max(all_nodes, key=lambda c: c.score if c.score is not None else float('-inf'))
            else:
                best_child = min(all_nodes, key=lambda c: c.score if c.score is not None else float('inf'))

            # if feedback_root.score is not None and best_child.score is not None:
            #     if (feedback_root.score > best_child.score) and bigger_is_better:
            #         MCTS_NODE_LIST = [root]
            #         return MCTS_NODE_LIST,root.workspace.file_dict
            #     if (feedback_root.score < best_child.score) and not bigger_is_better:
            #         MCTS_NODE_LIST = [root]
            #         return MCTS_NODE_LIST, root.workspace.file_dict
            
            if best_child is None:
                logger.warning("No child nodes expanded from root!")
                best_child = root
            logger.info(f"Best child visit count: {best_child.visit_count}, value: {best_child.value}")
            return MCTS_NODE_LIST,best_child.workspace.file_dict,best_child.feedback
        else:
            return  MCTS_NODE_LIST,root.workspace.file_dict,root.feedback



    def assign_code_list_to_evo(self, code_list: list[dict[str, str]], evo):
        """
        Assign the code list to the evolving item.

        The code list is aligned with the evolving item's sub-tasks.
        If a task is not implemented, put a None in the list.
        """
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                # evo.sub_workspace_list[index] = FBWorkspace(target_task=evo.sub_tasks[index])
                evo.sub_workspace_list[index] = evo.experiment_workspace
            if self.KEY_CHANGE_SUMMARY in code_list[index]:
                evo.sub_workspace_list[index].change_summary = code_list[index].pop(self.KEY_CHANGE_SUMMARY)
            evo.sub_workspace_list[index].inject_files(**code_list[index])
        return evo
    

class DSCoSTEERMCTSRunner(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:

        eval_l = [DSRunnerMCTSEvaluator(scen=scen)]
        if DS_RD_SETTING.enable_model_dump:
            eval_l.append(ModelDumpEvaluator(scen=scen, data_type="full"))

        eva = CoSTEERMultiEvaluator(
            single_evaluator=eval_l, scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        settings = DSRunnerMCSTCoSTEERSettings()
        es = DSRunnerMCTSMultiProcessEvolvingStrategy(scen=scen, settings=settings, max_iterations=DS_RD_SETTING.mcts_max_iterations, exploration_c=DS_RD_SETTING.mcts_exploration_constant)

        # In runner, we don't need very big loops, so we set max_loop to runner_max_loop
        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            max_loop=DS_RD_SETTING.runner_max_loop,
            **kwargs,
        )

    def get_develop_max_seconds(self) -> int | None:
        """
        The coder uses the scenario's real debug timeout as the maximum seconds for development.
        """
        #return 3600
        return int(self.scen.real_full_timeout() * self.settings.max_seconds_multiplier)

    def compare_and_pick_fb(self, base_fb: CoSTEERMultiFeedback | None, new_fb: CoSTEERMultiFeedback | None) -> bool:
        # In data science, we only have a single feedback.
        # Note: new_fb should always exists as indicated by _get_last_fb() function.
        if base_fb is None:
            return True

        base_fb = base_fb[0]
        new_fb = new_fb[0]

        def compare_scores(s1, s2) -> bool:
            if s2 is None:
                return False
            if s1 is None:
                return True
            return (s2 > s1) == self.scen.metric_direction

        return compare_scores(base_fb.score, new_fb.score)

    def develop(self, exp):
        bak_sub_tasks = exp.pending_tasks_list
        exp.sub_tasks = [
            CoSTEERTask(
                name="Debug running solution",
                description=f"You'll be provided with the source code and the running and testing stdout. "
                "Please check the error messages and debug the source code if any errors occur.\n"
                f"Original task: {bak_sub_tasks[0][0].get_task_information()}\n"
                f"Current code repo md5: {md5_hash(exp.experiment_workspace.all_codes)}",
            ),
        ]
        exp = super().develop(exp)  # run strategy(code implementation & evaluation loops)
        exp.sub_tasks = bak_sub_tasks

        # NOTE: after running the loops, we expect some results are generated
        #
        # 1) scores of the models and ensemble
        score_fp = exp.experiment_workspace.workspace_path / "scores.csv"
        if not score_fp.exists():
            logger.error("Metrics file (scores.csv) is not generated.")
            raise RunnerError(f"Metrics file (scores.csv) is not generated")
        exp.result = pd.read_csv(score_fp, index_col=0)
        exp.running_info.running_time = exp.experiment_workspace.running_info.running_time

        # 2) if mle-bench, then the submission format checking will be used.
        # DockerEnv for MLEBench submission validation
        if DS_RD_SETTING.if_using_mle_data:
            score_fp = exp.experiment_workspace.workspace_path / "test" / "mle_submission_format_test.output"
            with score_fp.open() as f:
                exp.format_check_result = f.read()
        return exp
