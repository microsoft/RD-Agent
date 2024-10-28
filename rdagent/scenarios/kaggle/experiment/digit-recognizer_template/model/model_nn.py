import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the neural network model with Batch Normalization
class NeuralNetwork(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=30, kernel_size=(3, 3), stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), stride=2)
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(30 * 6 * 6, 128)  # Adjust based on your input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    # Convert data to PyTorch tensors and reshape it for convolutional layers
    X_train_tensor = (
        torch.tensor(X_train.values, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    )  # Reshape and move to GPU
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.long).to(device)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    # Initialize the model, loss function and optimizer
    model = NeuralNetwork(input_channels=1, num_classes=len(set(y_train))).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Train the model
    num_epochs = 400
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                outputs = model(X_batch)
                valid_loss += criterion(outputs, y_batch).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}")

    return model


def predict(model, X):
    X_tensor = torch.tensor(X.values, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy().reshape(-1, 1)
