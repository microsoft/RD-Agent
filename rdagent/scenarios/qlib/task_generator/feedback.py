# TODO:
# Implement to feedback.

from rdagent.core.proposal import HypothesisExperiment2Feedback
from rdagent.core.proposal import Trace
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.oai.llm_utils import APIBackend
import json
import pandas as pd


class QlibFactorExperiment2Feedback(HypothesisExperiment2Feedback):
    def __init__(self):
        pass
    
    def generateFeedback(self, ti: Experiment, hypothesis: Hypothesis, baseline) -> HypothesisFeedback:
        """
        Generate feedback based on the executed experiment, hypothesis, and trace.
        Includes decision on whether to replace the SOTA result based on new results.
        """
        from rdagent.app.qlib_rd_loop.factor import ExperimentResults

        hypothesis_text = hypothesis.hypothesis
        current_result = ti.result
        tasks_factors = [(task.factor_name, task.factor_description) for task in ti.sub_tasks]

        last_experiment = baseline.last_experiment
        sota = baseline.sota
        alpha158 = baseline.alpha158
        print(ti.sub_tasks[0].factor_name)
        print(ti.sub_tasks[0].factor_description)
        print(ti.sub_tasks[0].factor_formulation)
        print(ti.sub_tasks[0].variables)

        # Generate the system prompt
        sys_prompt = (
            "You are a professional result analysis assistant. You will receive a hypothesis, multiple tasks with their factors, and some results. "
            "Your feedback should specify whether the current result supports or refutes the hypothesis, compare it with previous results, and suggest improvements or new directions."
            "Please provide detailed and constructive feedback. "
            "Example JSON Structure for Result Analysis: "
            '{"Observations": "Your overall observations here", "Feedback for Hypothesis": "Observations related to the hypothesis", '
            '"New Hypothesis": "Put your new hypothesis here.", "Reasoning": "Provide reasoning for the hypothesis here."}'
        )

        # Prepare task details
        task_details = "\n".join([f"Task: {factor_name}, Factor: {factor_description}" for factor_name, factor_description in tasks_factors])

        # Generate the user prompt
        usr_prompt = f'''
            We are in an experiment of finding hypotheses and validating or rejecting them so that in the end we have a powerful factor generated.
            Given the following hypothesis, tasks, factors, and current result, provide feedback on how well the result supports or refutes the hypothesis.
            Hypothesis: {hypothesis_text}\n
            Tasks and Factors:\n{task_details}\n
            Current Result: {current_result}\n
            Last Experiment Result: {last_experiment}\n
            SOTA Result: {sota}\n
            Alpha158 Result: {alpha158}\n
            Analyze the current result in the context of its ability to:
            1. Support or refute the hypothesis.
            2. Show improvement or deterioration compared to the last experiment.
            3. Outperform the SOTA result or not.
            4. Demonstrate positive or negative effects when compared to Alpha158.

            Provide detailed feedback and recommend whether to replace the SOTA if the new factor proves superior.
        '''

        try:
            # Call the APIBackend to generate the response for hypothesis feedback
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=usr_prompt,
                system_prompt=sys_prompt,
                json_mode=True,
            )

            # Parse the JSON response to extract the feedback
            response_json = json.loads(response)
            replace_sota = response_json.get("Replace SOTA", False)
            
            hypothesis_feedback = HypothesisFeedback(
                observations=response_json.get("Observations", "No observations provided"),
                feedback_for_hypothesis=response_json.get("Feedback for Hypothesis", "No feedback provided"),
                new_hypothesis=response_json.get("New Hypothesis", "No new hypothesis provided"),
                reasoning=response_json.get("Reasoning", "No reasoning provided"),
                replace_sota=replace_sota
            )

            print("Generated Hypothesis Feedback:\n", hypothesis_feedback)
            return hypothesis_feedback

        except json.JSONDecodeError as e:
            print("Error parsing JSON response from LLM for hypothesis feedback:", e)
        except Exception as e:
            print("An unexpected error occurred while generating hypothesis feedback:", e)
        return HypothesisFeedback()
