import json
from rdagent.oai.llm_utils import APIBackend

def generate_feedback(result, code, hypothesis):
    # Define the system prompt
    sys_prompt = (
        "You are a professional code review assistant. You will receive some code, a result, and a hypothesis. "
        "Your task is to provide feedback on how well the code and result support or refute the hypothesis. "
        "Please provide detailed and constructive feedback."
    )

    # Define the user prompt
    usr_prompt = (f'''
        "Given the following hypothesis, result, and code, provide feedback on how well the code and result support or refute the hypothesis. "
        "Hypothesis: {hypothesis}\n"
        "Result: {result}\n"
        "Code:\n```python\n{code}\n```\n"
        "Please provide detailed and constructive feedback."
    ''')

    try:
        # Call the APIBackend to generate the response
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        # Log the raw response for debugging
        print("Raw Response:\n", response)

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)
        feedback = response_json.get("feedback", "No feedback provided")

        print("Generated Feedback:\n", feedback)

        return feedback

    except json.JSONDecodeError as e:
        print("Error parsing JSON response from LLM:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)

def test_generate_feedback():
    result = "The model achieved an accuracy of 85% on the validation set."
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

    feedback = generate_feedback(result, code, hypothesis)
    print("Final Feedback:\n", feedback)

if __name__ == "__main__":
    test_generate_feedback()

