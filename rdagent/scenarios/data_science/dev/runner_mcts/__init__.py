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
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.data_science.share.eval import ModelDumpEvaluator
from rdagent.core.exception import RunnerError
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.data_science.dev.runner_mcts.eval import DSRunnerEvaluator
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
        self.untried_actions : list[dict] = []  # 待探索的候选修改

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0



class DSRunnerMCTSMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):

    def __init__(self, scen, settings, max_iterations=2, exploration_c=1.4):
        super().__init__(scen=scen, settings=settings)
        self.max_iterations = max_iterations
        self.exploration_c = exploration_c
        self.root = MCTSNode(workspace=scen.workspace)
        self.KEY_CHANGE_SUMMARY = "__change_summary__"

    @wait_retry(retry_n=5)       
    def generate_modifications(
        self,
        target_task: CoSTEERTask,
        workspace: FBWorkspace,
        prev_task_feedback: CoSTEERSingleFeedback,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        num_candidates: int = 3,  # 生成候选数量
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

        if prev_task_feedback.acceptable is False:
            system_prompt = T(".prompts:DSCoSTEER.system_debugger").r(
                task_desc=task_info,
                out_spec=PythonBatchEditOut.get_spec(with_del=False),
                diff_mode=self.settings.diff_mode,
            )
        else:
            system_prompt = T(".prompts:DSCoSTEER.system_refine").r(
                out_spec=PythonBatchEditOut.get_spec(with_del=False),
                diff_mode=self.settings.diff_mode,
            )

        session = APIBackend().build_chat_session(session_system_prompt=system_prompt)

        user_prompt = T(".prompts:DSCoSTEER.user").r(
            code=workspace.all_codes,
            change_summary=workspace.change_summary,
            feedback=prev_task_feedback,
            hyperparameter_tuning_suggestion=(
                prev_task_feedback.hyperparameter_tuning_suggestion if prev_task_feedback.acceptable else None
            ),
            queried_former_failed_knowledge=queried_former_failed_knowledge,
            num_candidates=num_candidates,
        )

        code_raw = session.build_chat_completion(user_prompt=user_prompt)

        # 尝试解析 JSON
        try:
            code_candidates = json.loads(code_raw)
        except Exception:
            # 如果解析失败，尝试用单候选 fallback
            if self.settings.diff_mode:
                single_candidate = PythonBatchPatchOut.extract_output(code_raw, prefix=workspace.workspace_path)
            else:
                single_candidate = PythonBatchEditOut.extract_output(code_raw)
            code_candidates = [single_candidate]

        # 过滤掉 workspace 中不存在的文件
        final_candidates = []
        for candidate in code_candidates:
            candidate_filtered = {k: v for k, v in candidate.items() if k in workspace.file_dict.keys()}
            final_candidates.append(candidate_filtered)

        logger.info(f"Generated {len(final_candidates)} candidate modifications.")
        return final_candidates



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
        prev_task_feedback: CoSTEERSingleFeedback,
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
                target_task, node.workspace, prev_task_feedback, queried_knowledge
            )
            node.untried_actions = modifications_list
            # 初始化 children 列表
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


    def simulate(self, node: MCTSNode, target_task, prev_task_feedback, gt_workspace, queried_knowledge):
        evaluator = DSRunnerEvaluator(scen=self.scen)
        logger.info("Simulating node with workspace files")
        feedback = evaluator.evaluate(target_task, node.workspace, gt_workspace, queried_knowledge)
        logger.info("Simulating node with workspace files")

        reward = feedback.score if feedback.score is not None else (-1 if not feedback.acceptable else 0)
        return reward

    
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
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:

        if prev_task_feedback is None:
            # if no prev_task_feedback, it is the first loop; we do not make any changes and goto evaluators directly.
            return {}

        root = MCTSNode(workspace=workspace)
        logger.info(f"Starting MCTS with max_iterations={self.max_iterations}")
        for _ in range(self.max_iterations):
            logger.info(f"Starting iteration {_+1}/{self.max_iterations}")
            node = self.select(root)
            new_node = self.expand(node, target_task, prev_task_feedback, queried_knowledge)
            logger.info(f"Expanded node. Current root visit count: {root.visit_count}")
            reward = self.simulate(new_node, target_task, prev_task_feedback, workspace, queried_knowledge)
            logger.info(f"Simulation reward: {reward}")
            self.backpropagate(new_node, reward)
            logger.info(f"Iteration completed. Current root visit count: {root.visit_count}")

        best_child = max(root.children, key=lambda c: c.visit_count)
        logger.info(f"Best child visit count: {best_child.visit_count}, value: {best_child.value}")
        return best_child.workspace.file_dict



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
    



class DSCoSTEERRunner(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:

        eval_l = [DSRunnerEvaluator(scen=scen)]
        if DS_RD_SETTING.enable_model_dump:
            eval_l.append(ModelDumpEvaluator(scen=scen, data_type="full"))

        eva = CoSTEERMultiEvaluator(
            single_evaluator=eval_l, scen=scen
        )  # Please specify whether you agree running your eva in parallel or not
        settings = DSRunnerMCSTCoSTEERSettings()
        es = DSRunnerMCTSMultiProcessEvolvingStrategy(scen=scen, settings=settings, max_iterations=2, exploration_c=1.4)

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
