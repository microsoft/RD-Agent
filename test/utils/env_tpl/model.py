import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1, act="LeakyReLU", rnn_type="LSTM"):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_input = nn.Dropout(0.05)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise NotImplementedError(f"RNN type {rnn_type} is not supported")
        self.fc = nn.Linear(hidden_dim, output_dim)
        if act == "LeakyReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        elif act == "SiLU":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(f"Activation function {act} is not supported")
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.drop_input(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
model_cls = Net