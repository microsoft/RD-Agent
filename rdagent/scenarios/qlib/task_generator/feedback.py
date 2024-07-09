# TODO:
# Implement to feedback.

from pathlib import Path
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import HypothesisExperiment2Feedback
from rdagent.core.proposal import Trace
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.oai.llm_utils import APIBackend
import json
import pandas as pd

feedback_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")

class QlibFactorExperiment2Feedback(HypothesisExperiment2Feedback):
    def __init__(self):
        pass
    
    def generateFeedback(self, ti: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback based on the executed experiment, hypothesis, and trace.
        Includes decision on whether to replace the SOTA result based on new results.
        """
        print("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = ti.result
        tasks_factors = [(task.factor_name, task.factor_description) for task in ti.sub_tasks]

        last_experiment = ti.result.last_experiment
        alpha158 = ti.result.alpha158

        print("Current Result:\n", current_result)
        print("Last Experiment Result:\n", last_experiment)
        print("Alpha158 Result:\n", alpha158)

        # Generate the system prompt
        sys_prompt = feedback_prompts["data_feedback_generation"]["system"]

        # Prepare task details
        task_details = "\n".join([f"Task: {factor_name}, Factor: {factor_description}" for factor_name, factor_description in tasks_factors])

        # Generate the user prompt
        usr_prompt = f'''
            We are conducting an experiment to validate or reject hypotheses, aiming to generate a powerful factor.
            Given the following hypothesis, tasks, factors, and current result, provide feedback on how well the result supports or refutes the hypothesis.
            Hypothesis: {hypothesis_text}\n
            Tasks and Factors:\n{task_details}\n
            Current Result: {current_result}\n
            Last Experiment Result: {last_experiment}\n
            Alpha158 Result: {alpha158}\n
            Analyze the current result in the context of its ability to:
            1. Support or refute the hypothesis.
            2. Show improvement or deterioration compared to the last experiment.
            3. Demonstrate positive or negative effects when compared to Alpha158.

            Provide detailed feedback and recommend whether to replace the best result if the new factor proves superior.
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
            
            # Extract fields from JSON response
            observations = response_json.get("Observations", "No observations provided")
            feedback_for_hypothesis = response_json.get("Feedback for Hypothesis", "No feedback provided")
            new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
            reasoning = response_json.get("Reasoning", "No reasoning provided")
            replace_best_result = response_json.get("Replace Best Result", "no").lower() == "yes"
            
            # Create HypothesisFeedback object
            hypothesis_feedback = HypothesisFeedback(
                observations=observations,
                feedback_for_hypothesis=feedback_for_hypothesis,
                new_hypothesis=new_hypothesis,
                reasoning=reasoning,
                attitude=replace_best_result
            )

            print("Generated Hypothesis Feedback:\n", hypothesis_feedback)
            return hypothesis_feedback

        except json.JSONDecodeError as e:
            print("Error parsing JSON response from LLM for hypothesis feedback:", e)
        except Exception as e:
            print("An unexpected error occurred while generating hypothesis feedback:", e)
        return HypothesisFeedback()
