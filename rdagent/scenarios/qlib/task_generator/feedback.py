# TODO:
# Implement to feedback.
import json
from rdagent.oai.llm_utils import APIBackend
from rdagent.core.proposal import HypothesisExperiment2Feedback, Trace, Hypothesis, HypothesisFeedback, Scenario
from rdagent.core.experiment import Experiment


class QlibFactorHypothesisExperiment2Feedback(HypothesisExperiment2Feedback): ...



class QlibModelHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    """Generated feedbacks on the hypothesis from **Executed** Implementations of different tasks & their comparisons with previous performances"""

    def generateFeedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        The `ti` should be executed and the results should be included, as well as the comparison between previous results (done by LLM).
        For example: `mlflow` of Qlib will be included.
        """

        # Define the system prompt for hypothesis feedback
        sys_prompt_hypothesis = (
            "You are a professional result analysis assistant. You will receive a result and a hypothesis. "
            "Your task is to provide feedback on how well the result supports or refutes the hypothesis by judging from the observation of performance increase or decrease. "
            "Please provide detailed and constructive feedback. "
            "Example JSON Structure for Result Analysis: "
            '{"Observations": "Your overall observations here", "Feedback for Hypothesis": "Observations related to the hypothesis", '
            '"New Hypothesis": "Put your new hypothesis here.", "Reasoning": "Provide reasoning for the hypothesis here.", '
            '"Decision": "True or False"}'
        )

        # Define the user prompt for hypothesis feedback
        context = trace.scen
        last_experiment_info = trace.get_last_experiment_info()

        if last_experiment_info:
            last_hypothesis, last_task, last_result = last_experiment_info
            last_info_str = f"Last Round Information:\nHypothesis: {last_hypothesis.hypothesis}\nTask: {last_task}\nResult: {last_result}\n"
        else:
            last_info_str = "This is the first round. No previous information available."

        usr_prompt_hypothesis = f'''
            We are in an experiment of finding hypothesis and validating or rejecting them so that in the end we have a powerful model generated.
            Here are the context: {context}. 
            {last_info_str}
            
            Now let's come to this round. You will receive the result and you will evaluate if the performance increases or decreases. 
            Hypothesis: {hypothesis.hypothesis}\n
            Relevant Reasoning: {hypothesis.reason}\n
            Result: {exp.result}\n

            Compare and observe. Which result has a better return and lower risk? If the performance increases， the hypothesis should be considered positive (working). 
            Hence, with the hypotheses, relevant reasonings, and results in mind (comparison), provide detailed and constructive feedback and suggest a new hypothesis. 
        '''

        try:
            # Call the APIBackend to generate the response for hypothesis feedback
            response_hypothesis = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=usr_prompt_hypothesis,
                system_prompt=sys_prompt_hypothesis,
                json_mode=True,
            )

            # Log the raw response for debugging
            print("Raw Response for Hypothesis Feedback:\n", response_hypothesis)

            # Parse the JSON response to extract the feedback
            response_json_hypothesis = json.loads(response_hypothesis)
            hypothesis_feedback = HypothesisFeedback(
                observations=response_json_hypothesis.get("Observations", "No observations provided"),
                hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
                new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
                reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
                decision=response_json_hypothesis.get("Decision", "false").lower() == "true"
            )

            return hypothesis_feedback

        except json.JSONDecodeError as e:
            # TODO:  (Xiao) I think raising a specific type of ERROR to make caller know sth bad has happend would be more reasonable
            print("Error parsing JSON response from LLM for hypothesis feedback:", e)
        except Exception as e:
            print("An unexpected error occurred while generating hypothesis feedback:", e)

        return HypothesisFeedback(
            observations="No observations",
            hypothesis_evaluation="No feedback",
            new_hypothesis="No new hypothesis",
            reason="No reasoning",
            decision=False
        )

