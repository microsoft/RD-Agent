MODEL_TYPE = "Graph"
BATCH_SIZE = 32
NUM_FEATURES = 10
NUM_TIMESTEPS = 4
NUM_EDGES = 20
INPUT_VALUE = 1.0
import torch
from torch_geometric.nn import GCNConv
from gt_model import model_cls as gt_model_cls
from gen_model import model_cls as gen_model_cls


def run_model(data_in,model_cls, param_init_value=1.0):
    (node_feature, edge_index,z,positions,batch) = data_in
    node_features = node_feature
    if MODEL_TYPE == "Tabular":
        input_shape = (BATCH_SIZE, NUM_FEATURES)
        m = model_cls(num_features=input_shape[1])
        data = torch.full(input_shape, INPUT_VALUE)
    elif MODEL_TYPE == "TimeSeries":
        input_shape = (BATCH_SIZE, NUM_FEATURES, NUM_TIMESTEPS)
        m = model_cls(num_features=input_shape[1], num_timesteps=input_shape[2])
        data = torch.full(input_shape, INPUT_VALUE)
    elif MODEL_TYPE == "Graph":
        # node_feature = torch.randn(BATCH_SIZE, NUM_FEATURES)
        # node_features = node_feature
        # edge_index = torch.randint(0, BATCH_SIZE, (2, NUM_EDGES))
        # TODO : Find beter way to handle different model types (e.g. use **kwargs list)
        if MODEL_NAME == "A-DGN":
            m = model_cls(in_channels=NUM_FEATURES)
        elif MODEL_NAME == "Dir-GNN":
            m = model_cls(GCNConv(in_channels=node_features.size(-1), out_channels=node_features.size(-1)))
        elif MODEL_NAME == "GPSConv":
            m = model_cls(channels=node_features.size(-1), conv=GCNConv(in_channels=node_features.size(-1), out_channels=node_features.size(-1)))
        elif MODEL_NAME == "LINKX":
            m = model_cls(
                num_nodes=node_features.size(0),
                in_channels=node_features.size(1),
                hidden_channels=node_features.size(1),
                out_channels=node_features.size(1),
                num_layers=1,
            )
        elif MODEL_NAME == "PMLP":
            m = model_cls(
                in_channels=node_features.size(1),
                hidden_channels=node_features.size(1),
                out_channels=node_features.size(1),
                num_layers=1,
            )
        elif MODEL_NAME == "ViSNet":
            m = model_cls()
 

        data = (node_feature, edge_index)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    # Initialize all parameters of `m` to `param_init_value`
    for _, param in m.named_parameters():
        param.data.fill_(param_init_value)

    # Execute the model
    if MODEL_TYPE == "Graph":
        if MODEL_NAME == "ViSNet":
            out = m(z, positions, batch)
        else:
            out = m(*data)
    else:
        out = m(data)

    execution_model_output = out.cpu().detach().numpy()
    execution_feedback_str = f"Execution successful, output tensor shape: {execution_model_output.shape}"
    return execution_model_output, execution_feedback_str



def get_data_conf(init_val):
    exec_config = {"model_eval_param_init": init_val}
    node_feature = torch.randn(BATCH_SIZE, NUM_FEATURES)
    edge_index = torch.randint(0, BATCH_SIZE, (2, NUM_EDGES))
    # 生成模拟数据
    # 1. z: 原子序数
    # 模拟 H₂O (H:1, O:8) 和 CH₄ (C:6, H:1)
    z = torch.tensor([1, 1, 8, 6, 1, 1, 1], dtype=torch.long)  # 7 个原子
    # 2. pos: 原子位置 (随机生成，单位：Å)
    positions = torch.rand(7, 3, dtype=torch.float32)  # 7 个原子的 3D 坐标
    # 3. batch: 批次索引
    # 前 3 个原子属于第 0 个分子，后 4 个原子属于第 1 个分子
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return (node_feature, edge_index,z,positions,batch), exec_config



round_n = 10

eval_pairs: list[tuple] = []

# run different input value
for _ in range(round_n):
    # run different model initial parameters.
    for init_val in [-0.2, -0.1, 0.1, 0.2]:
        (node_feature, edge_index,z,positions,batch), exec_config = get_data_conf(1.0)
        data_in = (node_feature, edge_index,z,positions,batch)
        gt_res, _ = run_model(data_in,gt_model_cls, param_init_value=init_val)
        res, _ = run_model(data_in,gen_model_cls, param_init_value=init_val)
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

