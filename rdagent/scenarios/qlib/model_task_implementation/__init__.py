import json
import os
from rdagent.oai.llm_utils import APIBackend
from rdagent.core.prompts import Prompts
from rdagent.core.task_generator import TaskGenerator
from rdagent.oai.llm_utils import APIBackend
import re
from pathlib import Path
from typing import Sequence

from jinja2 import Environment, StrictUndefined
from rdagent.components.task_implementation.model_implementation.model import (
    ModelExperiment,
    ModelImplementation,
    ModelTask,
)
from rdagent.core.prompts import Prompts
from rdagent.core.task_generator import TaskGenerator
from rdagent.oai.llm_utils import APIBackend

class QlibModelCodeWriter(TaskGenerator[ModelExperiment]):
    def generate(self, exp: ModelExperiment) -> ModelExperiment:
        mti_l = []
        for t in exp.sub_tasks:
            mti = ModelImplementation(t)
            mti.prepare()
            # Define the system prompt
            sys_prompt = (
                "You are a professional code writing assistant. You will receive some hypotheses and "
                "you will be in charge of writing code to execute them. Follow best practices and ensure the code is correct and efficient."
            )

            # Define the user prompt
            usr_prompt = ('''
                            "Generate a json file with two keys: [code] & [explanation]. "
                "Using the following information, write a Python code using PyTorch / Torch Geometric to implement the model. Generate the class Net(nn.Module) "
                "This model is in the quantitative investment field and has only one layer. "
                "Implement the model forward function based on the provided model formula information:\n"
                "Hypothesis: The data shows time-series quality.\n"
                "You must complete the forward function as far as you can."
                Example Output:
                        {
            "code": "import torch\nimport torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self, input_dim, output_dim=1, layers=(256,), act=\"LeakyReLU\"):\n        super(Net, self).__init__()\n        layers = [input_dim] + list(layers)\n        dnn_layers = []\n        drop_input = nn.Dropout(0.05)\n        dnn_layers.append(drop_input)\n        hidden_units = input_dim\n        for i, (_input_dim, hidden_units) in enumerate(zip(layers[:-1], layers[1:])):\n            fc = nn.Linear(_input_dim, hidden_units)\n            if act == \"LeakyReLU\":\n                activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)\n            elif act == \"SiLU\":\n                activation = nn.SiLU()\n            else:\n                raise NotImplementedError(f\"This type of input is not supported\")\n            bn = nn.BatchNorm1d(hidden_units)\n            seq = nn.Sequential(fc, bn, activation)\n            dnn_layers.append(seq)\n        drop_input = nn.Dropout(0.05)\n        dnn_layers.append(drop_input)\n        fc = nn.Linear(hidden_units, output_dim)\n        dnn_layers.append(fc)\n        self.dnn_layers = nn.ModuleList(dnn_layers)\n        self._weight_init()\n\n    def _weight_init(self):\n        for m in self.modules():\n            if isinstance(m, nn.Linear):\n                nn.init.kaiming_normal_(m.weight, a=0.1, mode=\"fan_in\", nonlinearity=\"leaky_relu\")\n\n    def forward(self, x):\n        cur_output = x\n        for i, now_layer in enumerate(self.dnn_layers):\n            cur_output = now_layer(cur_output)\n        return cur_output",
            "explanation": "This Python code defines a flexible neural network class 'Net' using PyTorch. It's designed to be modular with customizable input dimensions, output dimensions, layer configurations, and activation functions. The network begins with a dropout layer, followed by several hidden layers dynamically created based on the 'layers' tuple. Each hidden layer consists of a linear transformation, batch normalization, and an activation function, which can be either LeakyReLU or SiLU. Dropout layers are added before and after the sequence of hidden layers to prevent overfitting. The class also includes a custom weight initialization method, initializing weights using the Kaiming normal method, which is suited for layers followed by LeakyReLU activations. This model could be effectively used in quantitative investment strategies to process time-series data or for general deep learning tasks requiring robust feature extraction and nonlinear transformations."
                }

                '''
            )
            try:
                # Call the APIBackend to generate the response
                response = APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=usr_prompt,
                    system_prompt=sys_prompt,
                    json_mode=True,
                )

                # Log the raw response for debugging
                print("Raw Response:\n", response)

                # Parse the JSON response to extract the code
                response_json = json.loads(response)
                code = response_json.get("code", "No code generated")
                explanation = response_json.get("explanation", "No explanation provided")

                print("Generated Code:\n", code)
                print("\nExplanation:\n", explanation)

                # Write the generated code to model.py
                output_dir = '/home/v-xisenwang/RD-Agent/test/utils/env_tpl'
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, 'model.py')

                with open(output_file, 'w') as f:
                    f.write(code)
                print(f"Code has been written to {output_file}")

            except json.JSONDecodeError as e:
                print("Error parsing JSON response from LLM:", e)
            except Exception as e:
                print("An unexpected error occurred:", e)

            mti.inject_code(**{"model.py": code})
            mti_l.append(mti)
        exp.sub_implementations = mti_l
        return exp   

    def print_model_details():
        # Create an instance of APIBackend
        api_backend = APIBackend()

        # Check which model is being used
        if api_backend.use_llama2:
            print("Using LLaMA 2")
        elif api_backend.use_gcr_endpoint:
            print(f"Using GCR endpoint: {api_backend.gcr_endpoint}")
        else:
            print(f"Using Azure model: {api_backend.chat_model}")
