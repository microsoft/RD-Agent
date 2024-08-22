import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFeatureInteractionModel(nn.Module):
    def __init__(self, num_features):
        super(HybridFeatureInteractionModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128, momentum=0.1)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.1)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

model_cls = HybridFeatureInteractionModel