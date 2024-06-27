"""
This file will be removed in the future and replaced by
- rdagent/app/model_implementation/eval.py
"""
from dotenv import load_dotenv
from rdagent.oai.llm_utils import APIBackend

# randomly generate a input graph, node_feature and edge_index
# 1000 nodes, 128 dim node feature, 2000 edges
import torch
import os

assert load_dotenv()
formula_info = {
    "name": "Anti-Symmetric Deep Graph Network (A-DGN)",
    "description": "A framework for stable and non-dissipative DGN design. It ensures long-range information preservation between nodes and prevents gradient vanishing or explosion during training.",
    "formulation": "x_u^{(l)} = x_u^{(l-1)} + \\epsilon \\sigma \\left( W^T x_u^{(l-1)} + \\Phi(X^{(l-1)}, N_u) + b \\right)",
    "variables": {
        "x_u^{(l)}": "The state of node u at layer l",
        "\\epsilon": "The step size in the Euler discretization",
        "\\sigma": "A monotonically non-decreasing activation function",
        "W": "An anti-symmetric weight matrix",
        "X^{(l-1)}": "The node feature matrix at layer l-1",
        "N_u": "The set of neighbors of node u",
        "b": "A bias vector",
    },
}

system_prompt = "You are an assistant whose job is to answer user's question."
user_prompt = "With the following given information, write a python code using pytorch and torch_geometric to implement the model. This model is in the graph learning field, only have one layer. The input will be node_feature [num_nodes, dim_feature] and edge_index [2, num_edges], and they should be loaded from the files 'node_features.pt' and 'edge_index.pt'. There is not edge attribute or edge weight as input. The model should detect the node_feature and edge_index shape, if there is Linear transformation layer in the model, the input and output shape should be consistent. The in_channels is the dimension of the node features. You code should contain additional 'if __name__ == '__main__', where you should load the node_feature and edge_index from the files and run the model, and save the output to a file 'llm_output.pt'. Implement the model forward function based on the following information: model formula information. 1. model name: {}, 2. model description: {}, 3. model formulation: {}, 4. model variables: {}. You must complete the forward function as far as you can do.".format(
    formula_info["name"],
    formula_info["description"],
    formula_info["formulation"],
    formula_info["variables"],
)

resp = APIBackend(use_chat_cache=False).build_messages_and_create_chat_completion(
    user_prompt, system_prompt
)

print(resp)

# take the code part from the response and save it to a file, the code is covered in the ```python``` block
code = resp.split("```python")[1].split("```")[0]
with open("llm_code.py", "w") as f:
    f.write(code)

average_shape_eval = []
average_value_eval = []
for test_mode in ["zeros", "ones", "randn"]:

    if test_mode == "zeros":
        node_feature = torch.zeros(1000, 128)
    elif test_mode == "ones":
        node_feature = torch.ones(1000, 128)
    elif test_mode == "randn":
        node_feature = torch.randn(1000, 128)
    edge_index = torch.randint(0, 1000, (2, 2000))

    torch.save(node_feature, "node_features.pt")
    torch.save(edge_index, "edge_index.pt")

    try:
        os.system("python llm_code.py")
    except:
        print("Error in running the LLM code")
    os.system("python gt_code.py")
    os.system("rm edge_index.pt")
    os.system("rm node_features.pt")
    # load the output and print the shape

    from evaluator import shape_evaluator, value_evaluator

    try:
        llm_output = torch.load("llm_output.pt")
    except:
        llm_output = None
    gt_output = torch.load("gt_output.pt")

    average_shape_eval.append(shape_evaluator(llm_output, gt_output)[1])
    average_value_eval.append(value_evaluator(llm_output, gt_output)[1])

    print("Shape evaluation: ", average_shape_eval[-1])
    print("Value evaluation:super().generate(task_l) ", average_value_eval[-1])

    os.system("rm llm_output.pt")
    os.system("rm gt_output.pt")
os.system("rm llm_code.py")

print("Average shape evaluation: ", sum(average_shape_eval) / len(average_shape_eval))
print("Average value evaluation: ", sum(average_value_eval) / len(average_value_eval))
