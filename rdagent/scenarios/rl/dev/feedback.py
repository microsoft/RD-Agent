import json
from typing import Any

from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class RLExperiment2Feedback(Experiment2Feedback):
    """Generate feedback for RL post-training experiments using LLM."""

    def __init__(self, scen: Scenario, version: str = "exp_feedback") -> None:
        super().__init__(scen)
        self.version = version

    def generate_feedback(
        self, exp: Any, trace: Any | None = None, exception: Exception | None = None
    ) -> HypothesisFeedback:
        """Generate feedback using LLM."""
        # 获取实验结果
        result = getattr(exp, "result", {}) or {}
        exit_code = result.get("exit_code", -1)
        stdout = result.get("stdout", "")
        running_time = result.get("running_time", 0)
        benchmark = result.get("benchmark")
        benchmark_summary = None
        if benchmark:
            try:
                benchmark_summary = json.dumps(benchmark, ensure_ascii=False, indent=2)
            except TypeError:
                benchmark_summary = str(benchmark)
        
        # 获取假设和任务描述
        hypothesis = str(exp.hypothesis) if exp.hypothesis else "N/A"
        task_desc = exp.sub_tasks[0].get_task_information() if exp.sub_tasks else "N/A"
        
        if exception is not None:
            return self._gen_error_feedback(hypothesis, str(exception))
        
        return self._gen_feedback_with_llm(
            hypothesis=hypothesis,
            task_desc=task_desc,
            exit_code=exit_code,
            stdout=stdout,
            running_time=running_time,
            benchmark=benchmark_summary,
        )

    def _gen_feedback_with_llm(
        self,
        hypothesis: str,
        task_desc: str,
        exit_code: int,
        stdout: str,
        running_time: float,
        benchmark: str | None,
    ) -> HypothesisFeedback:
        """Generate feedback using LLM."""
        system_prompt = T(".prompts:exp_feedback.system").r()
        user_prompt = T(".prompts:exp_feedback.user").r(
            hypothesis=hypothesis,
            task_desc=task_desc,
            exit_code=exit_code,
            stdout=stdout,
            running_time=running_time,
            benchmark=benchmark,
            exception=None,
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
        )
        resp_dict = json.loads(resp)

        decision = resp_dict.get("decision", exit_code == 0)
        reason = resp_dict.get("reason", "")
        suggestions = resp_dict.get("suggestions", "")

        logger.info(f"Feedback: decision={decision}, reason={reason[:100]}...")

        return HypothesisFeedback(
            decision=decision,
            reason=reason,
            code_change_summary=suggestions,
        )

    def _gen_error_feedback(self, hypothesis: str, error_info: str) -> HypothesisFeedback:
        """Generate feedback for failed experiments."""
        system_prompt = T(".prompts:exp_feedback_error.system").r()
        user_prompt = T(".prompts:exp_feedback_error.user").r(
            hypothesis=hypothesis,
            error_info=error_info,
        )

        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
        )
        resp_dict = json.loads(resp)

        error_type = resp_dict.get("error_type", "Unknown")
        root_cause = resp_dict.get("root_cause", error_info)
        fix_suggestion = resp_dict.get("fix_suggestion", "")

        logger.error(f"Error feedback: {error_type} - {root_cause[:100]}...")

        return HypothesisFeedback(
            decision=False,
            reason=f"[{error_type}] {root_cause}",
            code_change_summary=fix_suggestion,
        )
