MODEL_TYPE = "Graph"
MODEL_NAME = "A-DGN"
BATCH_SIZE = 32
NUM_FEATURES = 10
NUM_TIMESTEPS = 4
NUM_EDGES = 20
INPUT_VALUE = 1.0
import torch
from gt_model import model_cls as gt_model_cls
from gen_model import model_cls as gen_model_cls


def run_model(data_in,model_cls, param_init_value=1.0):
    (node_features, edge_index,test_convolutional_input,test_RNN_and_LSTM_input,test_Seq2Seq_input,test_transformer_input,test_swin_transformer_input) = data_in
    if MODEL_TYPE == "Graph":
        if MODEL_NAME == "A-DGN":
            m = model_cls(in_channels=NUM_FEATURES)
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
        out = m(node_features, edge_index)
    elif MODEL_TYPE == "Convolutional":
        if MODEL_NAME == "AlexNet":
            m = model_cls()
        elif MODEL_NAME == "ResNet":
            m = model_cls()
        elif MODEL_NAME == "LeNet":
            m = model_cls()
        out = m(test_convolutional_input)
    elif MODEL_TYPE == "Recurrent":
        if MODEL_NAME == "RNN" or MODEL_NAME == "LSTM":
            m = model_cls(input_size=100,hidden_size=128,num_layers=2,bidirectional=True,batch_first=True)
            out, _  = m(test_RNN_and_LSTM_input)
        elif MODEL_TYPE == "Seq2SeqEncoder":
            m = model_cls(vocab_size=100, max_len=10, hidden_size=100)
            out, _ = m(test_Seq2Seq_input)
        elif MODEL_NAME == "Seq2SeqDecoder":
            m = model_cls(vocab_size=100, max_len=10, hidden_size=100, sos_id=0, eos_id=1)
            out, _, _ = m(test_Seq2Seq_input)
    elif MODEL_TYPE == "Transformer":
        if MODEL_NAME == "TransformerEncoder":
            m = model_cls()
            out = m(test_transformer_input)
        elif MODEL_NAME == "TransformerDecoder":
            m = model_cls()
            out = m(test_transformer_input,test_transformer_input)
        elif MODEL_NAME == "SwinTransformer":
            # TODO
            m = model_cls(patch_size=[4, 4],embed_dim=96,depths=[2, 2, 6, 2],num_heads=[3, 6, 12, 24],window_size=[7, 7],stochastic_depth_prob=0.2,num_classes=1000)
            out = m(test_swin_transformer_input)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    # Initialize all parameters of `m` to `param_init_value`
    for _, param in m.named_parameters():
        param.data.fill_(param_init_value)

    execution_model_output = out.cpu().detach().numpy()
    execution_feedback_str = f"Execution successful, output tensor shape: {execution_model_output.shape}"
    return execution_model_output, execution_feedback_str



def get_data_conf(init_val):
    exec_config = {"model_eval_param_init": init_val}
    node_features = torch.randn(BATCH_SIZE, NUM_FEATURES)
    edge_index = torch.randint(0, BATCH_SIZE, (2, NUM_EDGES))
    test_convolutional_input = torch.randn(1, 3, 224, 224)
    test_RNN_and_LSTM_input = torch.randn(3, 5, 100)
    test_Seq2Seq_input = torch.randint(low=0, high=100, size=(32, 10), dtype=torch.long)
    test_transformer_input = torch.randn(32, 10, 512) 
    test_swin_transformer_input = torch.randn(32, 3, 224, 224)
    return (node_features, edge_index,test_convolutional_input,test_RNN_and_LSTM_input,test_Seq2Seq_input,test_transformer_input,test_swin_transformer_input), exec_config



round_n = 10

eval_pairs: list[tuple] = []

# run different input value
for _ in range(round_n):
    # run different model initial parameters.
    for init_val in [-0.2, -0.1, 0.1, 0.2]:
        (node_features, edge_index,test_convolutional_input,test_RNN_and_LSTM_input,test_Seq2Seq_input,test_transformer_input,test_swin_transformer_input), exec_config = get_data_conf(1.0)
        data_in = (node_features, edge_index,test_convolutional_input,test_RNN_and_LSTM_input,test_Seq2Seq_input,test_transformer_input,test_swin_transformer_input)
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

