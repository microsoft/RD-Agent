import json
from rdagent.oai.llm_utils import APIBackend

class Task:
    def __init__(self, name: str, factor: str):
        self.name = name
        self.factor = factor

def generate_hypothesis_feedback(hypothesis, tasks, result, context):
    # Define the system prompt for hypothesis feedback
    sys_prompt_hypothesis = (
        "You are a professional result analysis assistant. You will receive a hypothesis, multiple tasks with their factors, and a result. "
        "Your task is to provide feedback on how well the result supports or refutes the hypothesis. "
        "Please provide detailed and constructive feedback. "
        "Example JSON Structure for Result Analysis: "
        '{"Observations": "Your overall observations here", "Feedback for Hypothesis": "Observations related to the hypothesis", '
        '"New Hypothesis": "Put your new hypothesis here.", "Reasoning": "Provide reasoning for the hypothesis here."}'
    )

    # Prepare task details
    task_details = "\n".join([f"Task: {task.name}, Factor: {task.factor}" for task in tasks])

    # Define the user prompt for hypothesis feedback
    usr_prompt_hypothesis = f'''
        We are in an experiment of finding hypotheses and validating or rejecting them so that in the end we have a powerful factor generated.
        Here is the context: {context}. 
        Now let's come to this round. 
        Given the following hypothesis, tasks, factors, and result, provide feedback on how well the result supports or refutes the hypothesis.
        Hypothesis: {hypothesis}\n
        Tasks and Factors:\n{task_details}\n
        Result: {result}\n
        Please provide detailed and constructive feedback.
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
        hypothesis_feedback = {
            "Observations": response_json_hypothesis.get("Observations", "No observations provided"),
            "Feedback for Hypothesis": response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            "New Hypothesis": response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            "Reasoning": response_json_hypothesis.get("Reasoning", "No reasoning provided"),
        }

        print("Generated Hypothesis Feedback:\n", hypothesis_feedback)

        return hypothesis_feedback

    except json.JSONDecodeError as e:
        print("Error parsing JSON response from LLM for hypothesis feedback:", e)
    except Exception as e:
        print("An unexpected error occurred while generating hypothesis feedback:", e)

# Example usage
if __name__ == "__main__":
    hypothesis = "The data shows time-series quality."
    tasks = [
        Task(name="momentum_10day", factor="momentum_10day"),
        Task(name="momentum_20day", factor="momentum_20day"),
    ]
    result = ''' Previous results are our standard benchmark, and the current result is the result of the new factors that follows the hypothesis and its tasks. 
    Comparison of Benchmark Return
Previous Results:
Mean: 0.0477%
Std: 1.2295%
Annualized Return: 11.3561%
Information Ratio: 0.5987
Max Drawdown: -37.0479%
Current Results:
Mean: 0.0161%
Std: 1.4384%
Annualized Return: 3.8237%
Information Ratio: 0.1723
Max Drawdown: -48.7617%
Observations:
The current benchmark return has lower mean and annualized returns.
Volatility (std) and max drawdown have increased, indicating higher risk.
Information ratio has significantly decreased, suggesting reduced efficiency.
Comparison of Excess Return Without Cost
Previous Results:
Mean: 0.0530%
Std: 0.5718%
Annualized Return: 12.6029%
Information Ratio: 1.4286
Max Drawdown: -7.2310%
Current Results:
Mean: 0.0265%
Std: 0.4131%
Annualized Return: 6.3043%
Information Ratio: 0.9892
Max Drawdown: -8.4109%
Observations:
Decrease in mean and annualized return.
Lower standard deviation, indicating reduced volatility.
Decreased information ratio, although still relatively high.
Slight increase in max drawdown.
Comparison of Excess Return With Cost
Previous Results:
Mean: 0.0339%
Std: 0.5717%
Annualized Return: 8.0654%
Information Ratio: 0.9145
Max Drawdown: -8.6083%
Current Results:
Mean: 0.0098%
Std: 0.4133%
Annualized Return: 2.3216%
Information Ratio: 0.3641
Max Drawdown: -12.0422% '''
    context = "This experiment is aimed at improving time-series prediction accuracy."

    feedback = generate_hypothesis_feedback(hypothesis, tasks, result, context)
    print("Final Feedback:\n", feedback)
