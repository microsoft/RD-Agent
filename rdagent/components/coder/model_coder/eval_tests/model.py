MODEL_TYPE = "Graph"
BATCH_SIZE = 32
NUM_FEATURES = 10
NUM_TIMESTEPS = 4
NUM_EDGES = 20
INPUT_VALUE = 1.0

import torch

from gt_model import model_cls as gt_model_cls
from gen_model import model_cls as gen_model_cls


def run_model(model_cls, param_init_value=1.0):
    if MODEL_TYPE == "Tabular":
        input_shape = (BATCH_SIZE, NUM_FEATURES)
        m = model_cls(num_features=input_shape[1])
        data = torch.full(input_shape, INPUT_VALUE)
    elif MODEL_TYPE == "TimeSeries":
        input_shape = (BATCH_SIZE, NUM_FEATURES, NUM_TIMESTEPS)
        m = model_cls(num_features=input_shape[1], num_timesteps=input_shape[2])
        data = torch.full(input_shape, INPUT_VALUE)
    elif MODEL_TYPE == "Graph":
        node_feature = torch.randn(BATCH_SIZE, NUM_FEATURES)
        edge_index = torch.randint(0, BATCH_SIZE, (2, NUM_EDGES))
        m = model_cls(in_channels=NUM_FEATURES)
        data = (node_feature, edge_index)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    # Initialize all parameters of `m` to `param_init_value`
    for _, param in m.named_parameters():
        param.data.fill_(param_init_value)

    # Execute the model
    if MODEL_TYPE == "Graph":
        out = m(*data)
    else:
        out = m(data)

    execution_model_output = out.cpu().detach().numpy()
    execution_feedback_str = f"Execution successful, output tensor shape: {execution_model_output.shape}"
    return execution_model_output, execution_feedback_str


gt_out = run_model(gt_model_cls)
gen_out = run_model(gen_model_cls)


def get_data_conf(init_val):
    in_dim = 1000
    in_channels = 128
    exec_config = {"model_eval_param_init": init_val}
    node_feature = torch.randn(in_dim, in_channels)
    edge_index = torch.randint(0, in_dim, (2, 2000))
    return (node_feature, edge_index), exec_config


round_n = 10

eval_pairs: list[tuple] = []

# run different input value
for _ in range(round_n):
    # run different model initial parameters.
    for init_val in [-0.2, -0.1, 0.1, 0.2]:
        gt_res, _ = run_model(gt_model_cls, param_init_value=init_val)
        res, _ = run_model(gen_model_cls, param_init_value=init_val)
        eval_pairs.append((res, gt_res))

# flat and concat the output
res_batch, gt_res_batch = [], []
for res, gt_res in eval_pairs:
    res_batch.append(torch.tensor(res).reshape(-1))
    gt_res_batch.append(torch.tensor(gt_res).reshape(-1))
res_batch = torch.stack(res_batch)
gt_res_batch = torch.stack(gt_res_batch)

res_batch = res_batch.detach().numpy()
gt_res_batch = gt_res_batch.detach().numpy()


# pearson correlation of each hidden output
def norm(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)

dim_corr = (norm(res_batch) * norm(gt_res_batch)).mean(axis=0)  # the correlation of each hidden output

# aggregate all the correlation
avr_corr = dim_corr.mean()
# FIXME:
# It is too high(e.g. 0.944) .
# Check if it is not a good evaluation!!
# Maybe all the same initial params will results in extreamly high correlation without regard to the model structure.
import pandas as pd
pd.Series({"result": avr_corr}).to_csv("result.csv")

