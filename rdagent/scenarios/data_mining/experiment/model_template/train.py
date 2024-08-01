import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import model_cls
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Set device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class MyDataset(Dataset):
    def __init__(self, x, label, device):
        self.x1 = x
        self.label = label
        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.FloatTensor(self.x1[idx]).to(self.device), torch.tensor(self.label[idx], dtype=torch.float).to(
            self.device
        )


def collate_fn(batch):
    x, label = [], []
    for data in batch:
        x.append(data[0])
        label.append(data[1])
    return torch.stack(x, 0), torch.stack(label, 0)


datapath = "/root/.data"
# datapath = '/home/v-suhancui/RD-Agent/physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3'


X = sparse.load_npz(datapath + "/features/ARF_12h/X.npz").todense()
df_pop = pd.read_csv(datapath + "/population/ARF_12h.csv")["ARF_LABEL"]

X = X.transpose(0, 2, 1)

indices = [i for i in range(len(df_pop))]
random.shuffle(indices)
split_point = int(0.7 * len(df_pop))

X_train, y_train = X[indices[:split_point]], np.array(df_pop[indices[:split_point]])
X_test, y_test = X[indices[split_point:]], np.array(df_pop[indices[split_point:]])


train_dataloader = DataLoader(
    MyDataset(X_train, y_train, device), collate_fn=collate_fn, shuffle=True, drop_last=True, batch_size=64
)
test_dataloader = DataLoader(
    MyDataset(X_test, y_test, device), collate_fn=collate_fn, shuffle=False, drop_last=False, batch_size=64
)

num_features = 4816
num_timesteps = 12
# Define the optimizer and loss function
model = model_cls(num_features=num_features, num_timesteps=num_timesteps).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


# Train the model
def eval_auc(model):
    y_pred = []
    for data in test_dataloader:
        x, y = data
        out = model(x)
        y_pred.append(out.cpu().detach().numpy())
    return roc_auc_score(y_test, np.concatenate(y_pred))


best = 0.0
best_model = None

for i in range(15):
    for data in train_dataloader:
        x, y = data
        out = model(x)
        optimizer.zero_grad()
        loss = criterion(out.squeeze(), y)
        loss.backward()
        optimizer.step()
    roc = eval_auc(model)
    if roc > best:
        best = roc
        best_model = model

y_pred = []
for data in test_dataloader:
    x, y = data
    out = best_model(x)
    y_pred.append(out.cpu().detach().numpy())

acc = roc_auc_score(y_test, np.concatenate(y_pred))

print(acc)

res = pd.Series(data=[acc], index=["AUROC"])
res.to_csv("./submission.csv")
