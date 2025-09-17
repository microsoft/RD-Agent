"""
Simplified Proposal Generator for Data Science Experiments

This module provides a streamlined approach to generate experiment proposals
by directly synthesizing tasks from feedback, bypassing complex hypothesis
generation and problem identification steps.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.pipeline.exp import PipelineTask
from rdagent.components.coder.data_science.workflow.exp import WorkflowTask
from rdagent.core.proposal import ExpGen
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.dev.feedback import ExperimentFeedback
from rdagent.scenarios.data_science.experiment.experiment import DSExperiment
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSHypothesis, DSTrace
from rdagent.scenarios.data_science.proposal.exp_gen.draft.draft import DSDraftExpGen
from rdagent.scenarios.data_science.proposal.exp_gen.planner import DSExperimentPlan
from rdagent.scenarios.data_science.proposal.exp_gen.proposal import draft_exp_in_decomposition
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import wait_retry


class TaskDesign(BaseModel):
    """Schema for task design output from LLM"""
    description: str = Field(description="Detailed description of the task implementation")
    key_improvements: List[str] = Field(
        description="List of specific improvements based on feedback",
        default_factory=list
    )
    workflow_update: Optional[str] = Field(
        description="Optional workflow update if needed",
        default=None
    )


class DSSimplifiedProposalExpGen(ExpGen):
    """
    Simplified proposal generator that directly converts feedback into tasks.

    This generator bypasses problem identification and hypothesis generation,
    creating a more streamlined path from feedback to actionable tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_response_schema = APIBackend().supports_response_schema()

    def gen(
        self,
        trace: DSTrace,
        plan: DSExperimentPlan | None = None,
    ) -> DSExperiment:
        """
        Generate an experiment directly from trace feedback.

        Args:
            trace: The experiment trace containing feedback history
            plan: Optional experiment plan (currently unused)

        Returns:
            A new DSExperiment with tasks generated from feedback
        """
        # Check pipeline mode setting
        pipeline = DS_RD_SETTING.coder_on_whole_pipeline

        # Handle first-round decomposition for non-pipeline mode
        if not pipeline and (draft_exp := draft_exp_in_decomposition(self.scen, trace)):
            logger.info("Using draft decomposition for initial component")
            return draft_exp

        # Get SOTA experiment and feedback
        sota_exp, fb_to_sota_exp = self._get_sota_and_feedback(trace)

        # Prepare context descriptions
        scenario_desc = self._get_scenario_description(sota_exp)
        sota_exp_desc = self._get_sota_description(sota_exp)

        # Get feedback summaries
        all_feedback = trace.experiment_and_feedback_list_after_init(return_type="all")
        failed_feedback = trace.experiment_and_feedback_list_after_init(return_type="failed")

        # Check if this is the first round
        is_first_round = len(trace.hist) == 0

        # Generate task directly from feedback
        logger.info(f"Generating task from feedback (first_round={is_first_round})")
        task_design, workflow_update = self._direct_feedback_to_task(
            scenario_desc=scenario_desc,
            sota_exp_desc=sota_exp_desc,
            sota_exp=sota_exp,
            all_feedback=all_feedback,
            failed_feedback=failed_feedback,
            fb_to_sota_exp=fb_to_sota_exp,
            is_first_round=is_first_round,
            pipeline=pipeline,
        )

        # Create experiment with generated task
        return self._create_experiment(
            task_design=task_design,
            workflow_update=workflow_update,
            sota_exp=sota_exp,
            pipeline=pipeline,
        )

    def _get_sota_and_feedback(self, trace: DSTrace) -> Tuple[Optional[DSExperiment], Optional[ExperimentFeedback]]:
        """Extract SOTA experiment and its feedback from trace."""
        if (sota_exp_fb := trace.sota_experiment_fb()) is None:
            return None, None
        return sota_exp_fb

    def _get_scenario_description(self, sota_exp: Optional[DSExperiment]) -> str:
        """Get scenario description with optional EDA output."""
        if not isinstance(sota_exp, DSExperiment):
            eda_output = None
        else:
            eda_output = sota_exp.experiment_workspace.file_dict.get("EDA.md", None)
        return self.scen.get_scenario_all_desc(eda_output=eda_output)

    def _get_sota_description(self, sota_exp: Optional[DSExperiment]) -> str:
        """Get SOTA experiment description."""
        return T("scenarios.data_science.share:describe.exp").r(
            exp=sota_exp,
            heading="Current best implementation"
        )

    @wait_retry(retry_n=3)
    def _direct_feedback_to_task(
        self,
        scenario_desc: str,
        sota_exp_desc: str,
        sota_exp: Optional[DSExperiment],
        all_feedback: List,
        failed_feedback: List,
        fb_to_sota_exp: Optional[ExperimentFeedback],
        is_first_round: bool,
        pipeline: bool,
    ) -> Tuple[str, Optional[str]]:
        """
        Generate task directly from feedback using a single LLM call.

        This is the core simplification - we directly ask the LLM to synthesize
        an improvement plan based on feedback, without intermediate hypothesis steps.
        """

        # Build appropriate prompt based on round
        if is_first_round:
            system_prompt = self._build_initial_system_prompt(scenario_desc, pipeline)
            user_prompt = self._build_initial_user_prompt(scenario_desc)
        else:
            system_prompt = self._build_improvement_system_prompt(scenario_desc, pipeline)
            user_prompt = self._build_improvement_user_prompt(
                sota_exp_desc=sota_exp_desc,
                all_feedback=all_feedback,
                failed_feedback=failed_feedback,
                fb_to_sota_exp=fb_to_sota_exp,
            )

        # Single LLM call to generate task
        logger.info("Making simplified LLM call for task generation")

        if self.supports_response_schema:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                response_format=TaskDesign,
            )
            task_design = TaskDesign(**json.loads(response))
            return task_design.description, task_design.workflow_update
        else:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_target_type=Dict[str, str]
            )
            result = json.loads(response)
            return result.get("description", "Task description not provided"), result.get("workflow_update")

    def _build_initial_system_prompt(self, scenario_desc: str, pipeline: bool) -> str:
        """Build system prompt for first round (no feedback available)."""
        return f"""You are an expert data scientist creating an initial solution for a Kaggle competition.

Competition Details:
{scenario_desc}

Your task is to design a complete {'pipeline' if pipeline else 'initial approach'} that:
1. Loads and preprocesses the data appropriately
2. Implements a reasonable baseline model
3. Handles evaluation according to the competition metric
4. Produces valid submissions

Focus on creating a solid foundation that can be iteratively improved.
Provide a detailed implementation plan in the response."""

    def _build_initial_user_prompt(self, scenario_desc: str) -> str:
        """Build user prompt for first round."""
        return f"""Design an initial implementation for this competition.

Create a comprehensive plan that includes:
- Data loading and validation
- Basic preprocessing steps
- A reasonable baseline model choice
- Evaluation strategy
- Submission generation

Output a JSON with:
- "description": Detailed implementation plan (markdown format)
- "workflow_update": Optional workflow adjustments if needed"""

    def _build_improvement_system_prompt(self, scenario_desc: str, pipeline: bool) -> str:
        """Build system prompt for improvement rounds (with feedback)."""
        return f"""You are an expert data scientist improving a Kaggle competition solution based on feedback.

Competition Details:
{scenario_desc}

You will receive:
1. Current best implementation details
2. Feedback from previous experiments
3. Failed attempts and their issues

Your task is to synthesize this feedback into a concrete improvement plan.
Focus on actionable changes that directly address the identified issues.
Be specific about what to modify and why."""

    def _build_improvement_user_prompt(
        self,
        sota_exp_desc: str,
        all_feedback: List,
        failed_feedback: List,
        fb_to_sota_exp: Optional[ExperimentFeedback],
    ) -> str:
        """Build user prompt for improvement rounds."""
        # Format feedback summaries
        all_feedback_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=all_feedback,
            type="all",
            pipeline=True,
        )

        failed_feedback_desc = T("scenarios.data_science.share:describe.trace").r(
            exp_and_feedback_list=failed_feedback,
            type="failed",
            pipeline=True,
        )

        prompt = f"""Based on the feedback below, generate an improved implementation plan.

## Current Best Implementation
{sota_exp_desc}

## All Experiment History
{all_feedback_desc}

## Failed Attempts
{failed_feedback_desc}"""

        if fb_to_sota_exp and hasattr(fb_to_sota_exp, 'eda_improvement') and fb_to_sota_exp.eda_improvement:
            prompt += f"\n\n## EDA-based Improvements\n{fb_to_sota_exp.eda_improvement}"

        prompt += """

Analyze the feedback and create a concrete improvement plan that:
1. Addresses the main issues identified in feedback
2. Builds upon successful strategies
3. Avoids repeating failed approaches
4. Includes specific implementation details

Output a JSON with:
- "description": Detailed improvement plan (markdown format)
- "key_improvements": List of specific changes based on feedback
- "workflow_update": Optional workflow adjustments if needed"""

        return prompt

    def _create_experiment(
        self,
        task_design: str,
        workflow_update: Optional[str],
        sota_exp: Optional[DSExperiment],
        pipeline: bool,
    ) -> DSExperiment:
        """Create DSExperiment from generated task design."""

        # Create simplified hypothesis (for compatibility)
        hypothesis = DSHypothesis(
            component="Pipeline" if pipeline else "Model",
            hypothesis="Direct synthesis from feedback analysis",
            reason="Simplified proposal generation based on feedback",
        )

        # Create main task
        task = PipelineTask(
            name="Pipeline",
            description=task_design,
        )

        # Create experiment
        exp = DSExperiment(
            pending_tasks_list=[[task]],
            hypothesis=hypothesis,
        )

        # Inject SOTA code if available
        if sota_exp is not None:
            exp.experiment_workspace.inject_code_from_file_dict(sota_exp.experiment_workspace)

        # Add workflow update if needed
        if workflow_update and workflow_update != "No update needed":
            workflow_task = WorkflowTask(
                name="Workflow",
                description=workflow_update,
            )
            exp.pending_tasks_list.append([workflow_task])
            logger.info("Added workflow update task")

        logger.info(f"Created experiment with {len(exp.pending_tasks_list)} task(s)")
        return exp