## This is a dummy file to be replaced & injected

import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentAnalysisModel(nn.Module):
    def __init__(self, num_features, num_timesteps):
        super(SentimentAnalysisModel, self).__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=128, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)  # Added dropout for regularization
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Rearrange to (batch, num_timesteps, num_features)
        x, _ = self.gru(x)
        x = self.dropout(x[:,-1,:])  # Applying dropout to the output of the last timestep
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_cls = SentimentAnalysisModel
