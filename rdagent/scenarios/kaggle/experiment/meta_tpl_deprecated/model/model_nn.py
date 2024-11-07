import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Restored three-layer model structure
class FeatureInteractionModel(nn.Module):
    def __init__(self, num_features):
        super(FeatureInteractionModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


# Training function
def fit(X_train, y_train, X_valid, y_valid):
    num_features = X_train.shape[1]
    model = FeatureInteractionModel(num_features).to(device)
    criterion = nn.BCELoss()  # Binary classification problem
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert to TensorDataset and create DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train.to_numpy(), dtype=torch.float32), torch.tensor(y_train.reshape(-1), dtype=torch.float32)
    )
    valid_dataset = TensorDataset(
        torch.tensor(X_valid.to_numpy(), dtype=torch.float32), torch.tensor(y_valid.reshape(-1), dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Train the model
    model.train()
    for epoch in range(5):
        print(f"Epoch {epoch + 1}/5")
        epoch_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to the device
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(1)  # Reshape outputs to [32]
            loss = criterion(outputs, y_batch)  # Adjust target shape
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"End of epoch {epoch + 1}, Avg Loss: {epoch_loss / len(train_loader):.4f}")

    return model


# Prediction function
def predict(model, X):
    model.eval()
    predictions = []
    with torch.no_grad():
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)  # Move data to the device
        for i in tqdm(range(0, len(X_tensor), 32), desc="Predicting", leave=False):
            batch = X_tensor[i : i + 32]  # Predict in batches
            pred = model(batch).squeeze().cpu().numpy()  # Move results back to CPU
            predictions.extend(pred)
    return np.array(predictions)  # Return boolean predictions
