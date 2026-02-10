"""RL CoSTEER - Code generation component for RL post-training"""

from typing import Generator

from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator, CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERQueriedKnowledge
from rdagent.core.evolving_agent import EvolvingStrategy, EvoStep
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.log import rdagent_logger as logger


class RLCoderCoSTEERSettings(CoSTEERSettings):
    """RL Coder settings."""
    pass


class RLEvolvingStrategy(EvolvingStrategy):
    """RL code generation strategy using LLM."""

    def __init__(self, scen: Scenario, settings: CoSTEERSettings):
        self.scen = scen
        self.settings = settings

    def evolve_iter(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        evolving_trace: list[EvoStep] = [],
        **kwargs,
    ) -> Generator[EvolvingItem, EvolvingItem, None]:
        """Generate code for all tasks using LLM."""
        for index, target_task in enumerate(evo.sub_tasks):
            code = self._generate_code(target_task, evolving_trace)
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace
            evo.sub_workspace_list[index].inject_files(**code)

        evo = yield evo
        return

    def _generate_code(self, task: Task, evolving_trace: list[EvoStep] = []) -> dict[str, str]:
        """Generate RL training code using LLM."""
        from rdagent.app.rl.conf import RL_RD_SETTING

        # 获取上轮反馈
        feedback = None
        if evolving_trace:
            last_step = evolving_trace[-1]
            if hasattr(last_step, 'feedback') and last_step.feedback:
                feedback = str(last_step.feedback)

        # 构造 prompt
        system_prompt = T(".prompts:rl_coder.system").r()
        user_prompt = T(".prompts:rl_coder.user").r(
            task_description=task.description if hasattr(task, 'description') else str(task),
            base_model=RL_RD_SETTING.base_model or "",
            benchmark=RL_RD_SETTING.benchmark or "",
            hypothesis=str(task.name) if hasattr(task, 'name') else "Train RL model",
            feedback=feedback,
        )

        # 调用 LLM
        session = APIBackend().build_chat_session(session_system_prompt=system_prompt)
        code = session.build_chat_completion(
            user_prompt=user_prompt,
            json_mode=False,
            code_block_language="python",
        )
        logger.info(f"LLM generated code:\n{code[:200]}...")
        return {"main.py": code}

    def _mock_code(self) -> dict[str, str]:
        """Fallback mock code."""
        return {"main.py": '''import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("ppo_cartpole")
print("Training completed!")
'''}


class RLCoderEvaluator:
    """RL code evaluator (mock implementation)."""

    def __init__(self, scen: Scenario) -> None:
        self.scen = scen

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace | None,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
    ) -> CoSTEERSingleFeedback:
        """Evaluate RL code. Currently returns mock success."""
        # TODO: 实现真正的评估逻辑
        return CoSTEERSingleFeedback(
            execution="Mock: executed successfully",
            return_checking=None,
            code="Mock: code looks good",
            final_decision=True,
        )


class RLCoSTEER(CoSTEER):
    """RL CoSTEER - orchestrates code generation and evaluation."""

    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        settings = RLCoderCoSTEERSettings()
        eva = CoSTEERMultiEvaluator([RLCoderEvaluator(scen=scen)], scen=scen)
        es = RLEvolvingStrategy(scen=scen, settings=settings)

        super().__init__(
            *args,
            settings=settings,
            eva=eva,
            es=es,
            scen=scen,
            max_loop=1,
            stop_eval_chain_on_fail=False,
            with_knowledge=False,
            knowledge_self_gen=False,
            **kwargs,
        )
