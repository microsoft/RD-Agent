import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modified model for multi-class classification
class FeatureInteractionModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FeatureInteractionModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)  # Output nodes equal to num_classes
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)  # Apply softmax to get probabilities


# Training function
def fit(X_train, y_train, X_valid, y_valid):
    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))  # Determine number of classes
    model = FeatureInteractionModel(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert to TensorDataset and create DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train.to_numpy(), dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.long),  # Use long for labels
    )
    valid_dataset = TensorDataset(
        torch.tensor(X_valid.to_numpy(), dtype=torch.float32), torch.tensor(y_valid.to_numpy(), dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Train the model
    model.train()
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        epoch_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"End of epoch {epoch + 1}, Avg Loss: {epoch_loss / len(train_loader):.4f}")

    return model


# Prediction function
def predict(model, X):
    model.eval()
    probabilities = []
    with torch.no_grad():
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        for i in tqdm(range(0, len(X_tensor), 32), desc="Predicting", leave=False):
            batch = X_tensor[i : i + 32]
            pred = model(batch)
            probabilities.append(pred.cpu().numpy())  # Collect probabilities
    return np.vstack(probabilities)  # Return as a 2D array
