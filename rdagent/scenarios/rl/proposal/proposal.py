import json

from rdagent.app.rl.conf import RL_RD_SETTING
from rdagent.core.proposal import ExpGen, Hypothesis, Trace
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.rl.autorl_bench.tasks import RLTask
from rdagent.scenarios.rl.experiment.experiment import RLExperiment
from rdagent.utils.agent.tpl import T



class RLPostTrainingExpGen(ExpGen):
    """RL post-training experiment generator with LLM."""

    def __init__(self, scen: Scenario | None = None):
        super().__init__(scen)

    def gen(self, trace: Trace) -> RLExperiment:
        """Generate RL post-training experiment using LLM."""
        # 构建历史摘要
        trace_summary = self._build_trace_summary(trace)

        # 调用 LLM 生成假设
        hypothesis_data = self._gen_hypothesis_with_llm(trace_summary)

        # 创建任务和实验
        rl_task = RLTask(
            name=f"RLTask_{hypothesis_data.get('algorithm', 'PPO')}",
            description=hypothesis_data.get("hypothesis", "Train RL agent"),
        )
        hypothesis = Hypothesis(
            hypothesis=hypothesis_data.get("hypothesis", "Train RL agent"),
            reason=hypothesis_data.get("reason", ""),
            concise_reason="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
        )
        algorithm = hypothesis_data.get("algorithm", "PPO")
        exp = RLExperiment(sub_tasks=[rl_task], hypothesis=hypothesis)
        logger.info(f"Generated experiment: {hypothesis.hypothesis} (algorithm={algorithm})")
        return exp

    def _build_trace_summary(self, trace: Trace) -> str:
        """Build summary of historical experiments."""
        if not trace or not trace.hist:
            return ""
        
        summaries = []
        for i, (exp, feedback) in enumerate(trace.hist[-3:]):  # 最近3个实验
            status = "成功" if feedback is not None and feedback.decision else "失败"
            hypothesis = exp.hypothesis.hypothesis if exp.hypothesis else "N/A"
            summaries.append(f"### 实验{i+1}: {hypothesis}")
            summaries.append(f"- 结果: {status}")
            # 添加失败原因和建议
            if feedback is not None:
                if getattr(feedback, 'reason', None):
                    summaries.append(f"- 原因: {feedback.reason}")
                if getattr(feedback, 'code_change_summary', None):
                    summaries.append(f"- 建议: {feedback.code_change_summary}")
        
        return "\n".join(summaries)

    def _gen_hypothesis_with_llm(self, trace_summary: str) -> dict:
        """Generate hypothesis using LLM."""
        system_prompt = T(".prompts:hypothesis_gen.system").r()
        user_prompt = T(".prompts:hypothesis_gen.user").r(
            base_model=RL_RD_SETTING.base_model or "",
            trace_summary=trace_summary,
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
        )
        return json.loads(resp)
