import json
from typing import List, Optional

from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.core.proposal import ExpGen
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis
from rdagent.utils.agent.tpl import T


class FinetuneExpGen(ExpGen):
    """LLM Fine-tuning specific experiment generator.

    Focuses on LLM fine-tuning improvements rather than general data science workflows.
    Generates experiments based on fine-tuning specific challenges and optimizations.
    """

    def gen(self, trace=None) -> DSExperiment:
        """Generate LLM fine-tuning experiment based on current scenario and trace history."""

        # Get scenario description
        scenario_desc = self.scen.get_scenario_all_desc(eda_output=None)

        # Analyze previous experiments if trace exists
        if trace and len(trace.hist) > 0:
            return self._gen_iterative_experiment(scenario_desc, trace)
        else:
            return self._gen_initial_experiment(scenario_desc)

    def _gen_initial_experiment(self, scenario_desc: str) -> DSExperiment:
        """Generate the first experiment with baseline fine-tuning setup."""

        hypothesis = DSHypothesis(
            component="Pipeline",
            hypothesis="Implement baseline LLM fine-tuning using LLaMA-Factory with standard QLoRA configuration",
            reason="Initial experiment to establish baseline performance using efficient fine-tuning methods",
        )

        task_desc = self._generate_initial_task_description(scenario_desc)

        task = PipelineTask(
            name="LLM_Finetune_Baseline",
            description=task_desc,
        )

        return DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)

    def _gen_iterative_experiment(self, scenario_desc: str, trace) -> DSExperiment:
        """Generate improved experiment based on previous results and feedback."""

        # Get previous experiment results and feedback
        prev_experiments_desc = self._describe_previous_experiments(trace)

        # Generate hypothesis based on analysis
        hypothesis = self._generate_hypothesis(scenario_desc, prev_experiments_desc)

        # Generate task description
        task_desc = self._generate_iterative_task_description(scenario_desc, prev_experiments_desc, hypothesis)

        task = PipelineTask(
            name=f"LLM_Finetune_{hypothesis.component}_Optimization",
            description=task_desc,
        )

        return DSExperiment(pending_tasks_list=[[task]], hypothesis=hypothesis)

    def _describe_previous_experiments(self, trace) -> str:
        """Analyze previous experiments and extract key insights."""

        experiments_summary = []
        for exp, feedback in trace.hist[-3:]:  # Last 3 experiments
            if exp and feedback:
                experiments_summary.append(
                    f"Experiment: {exp.hypothesis.hypothesis if exp.hypothesis else 'Unknown'}\n"
                    f"Success: {feedback.decision}\n"
                    f"Feedback: {feedback.final_feedback}\n"
                )

        return "\n".join(experiments_summary) if experiments_summary else "No previous experiments"

    def _generate_hypothesis(self, scenario_desc: str, prev_experiments_desc: str) -> DSHypothesis:
        """Generate hypothesis for next iteration based on scenario and previous results."""

        sys_prompt = T(".prompts:llm_finetune_hypothesis_gen.system").r()
        user_prompt = T(".prompts:llm_finetune_hypothesis_gen.user").r(
            scenario_desc=scenario_desc,
            prev_experiments_desc=prev_experiments_desc,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        try:
            hypothesis_data = json.loads(response)
            return DSHypothesis(
                component=hypothesis_data.get("component", "Pipeline"),
                hypothesis=hypothesis_data.get("hypothesis", "Improve fine-tuning configuration"),
                reason=hypothesis_data.get("reason", "Based on previous experiment analysis"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse hypothesis generation response: {e}")
            return DSHypothesis(
                component="Pipeline",
                hypothesis="Optimize fine-tuning hyperparameters and configuration",
                reason="Default hypothesis due to parsing error",
            )

    def _generate_initial_task_description(self, scenario_desc: str) -> str:
        """Generate task description for initial baseline experiment."""

        sys_prompt = T(".prompts:llm_finetune_initial_task.system").r()
        user_prompt = T(".prompts:llm_finetune_initial_task.user").r(
            scenario_desc=scenario_desc,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
        )

        return response

    def _generate_iterative_task_description(
        self, scenario_desc: str, prev_experiments_desc: str, hypothesis: DSHypothesis
    ) -> str:
        """Generate task description for iterative improvement experiment."""

        sys_prompt = T(".prompts:llm_finetune_iterative_task.system").r()
        user_prompt = T(".prompts:llm_finetune_iterative_task.user").r(
            scenario_desc=scenario_desc,
            prev_experiments_desc=prev_experiments_desc,
            hypothesis=hypothesis.hypothesis,
            component=hypothesis.component,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
        )

        return response
