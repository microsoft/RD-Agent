import json
from rdagent.oai.llm_utils import APIBackend

def better_model_generate(feedback, hypothesis, code):
    # Define the system prompt for generating better code
    sys_prompt_better_code = (
        "You are a professional code improvement assistant. You will receive a hypothesis, initial code, and feedback on that code. "
        "Your task is to generate an improved version of the code based on the feedback, following best practices and ensuring the code is correct and efficient."
        "Example JSON Structure for Improved Code: "
        '{"code": "Your improved code here", "explanation": "Detailed explanation of the improvements made"}'
        'Remember, you must generate a class called Net'
    )

    # Define the user prompt for generating better code
    usr_prompt_better_code = f'''
        "Given the following hypothesis, initial code, and feedback, provide an improved version of the code based on the feedback."
        "Hypothesis: {hypothesis}\n"
        "Initial Code:\n```python\n{code}\n```\n"
        "Feedback: {feedback}\n"
        "Please provide an improved version of the code and explain the improvements made."
    '''

    try:
        # Call the APIBackend to generate the response for improved code
        response_better_code = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt_better_code,
            system_prompt=sys_prompt_better_code,
            json_mode=True,
        )

        # Log the raw response for debugging
        print("Raw Response for Improved Code:\n", response_better_code)

        # Parse the JSON response to extract the improved code and explanation
        response_json_better_code = json.loads(response_better_code)
        improved_code = response_json_better_code.get("code", "No code generated")
        explanation = response_json_better_code.get("explanation", "No explanation provided")

        print("Generated Improved Code:\n", improved_code)
        print("\nExplanation:\n", explanation)

        return improved_code, explanation

    except json.JSONDecodeError as e:
        print("Error parsing JSON response from LLM for improved code:", e)
    except Exception as e:
        print("An unexpected error occurred while generating improved code:", e)

# Example usage
if __name__ == "__main__":
    feedback = """
    To make the model more suitable for analyzing time-series data and supporting the hypothesis that 'the data shows time-series quality', consider the following modifications:
    1. Integrate RNN, LSTM, or GRU layers: These layers are specifically designed to handle sequential data and can capture temporal dependencies in a time-series. Adding one or more of these layers could enable the model to learn from the temporal structure of the data.
    2. Utilize 1D Convolutional Layers: 1D CNNs can be effective in identifying temporal patterns in time-series data when configured properly. They can serve as feature extractors that highlight important time-dependent characteristics in the data.
    3. Implement Time-Series Specific Preprocessing: Depending on the nature of the time-series, it might be beneficial to include preprocessing steps such as detrending, normalization, or differencing to make the series stationary. This could improve model performance by simplifying the patterns the network needs to learn.
    4. Explore Time-Distributed Layers: For deep learning models that involve multiple sequential or convolutional layers, applying the same operation at each time step (time-distributed processing) can be beneficial for capturing complex temporal relationships.
    Overall, adapting the model to incorporate some of these suggestions could enhance its ability to analyze and make predictions based on time-series data, thus better supporting the hypothesis.
    """

    code = '''
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, output_dim=1, act="LeakyReLU"):
        super(Net, self).__init__()
        self.drop_input = nn.Dropout(0.05)
        self.fc = nn.Linear(input_dim, output_dim)
        if act == "LeakyReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        elif act == "SiLU":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation function {act} is not supported")
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.drop_input(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    '''

    hypothesis = "The data shows time-series quality."

    improved_code, explanation = better_model_generate(feedback, hypothesis, code)
    print("Final Improved Code:\n", improved_code)
    print("Explanation of Improvements:\n", explanation)
