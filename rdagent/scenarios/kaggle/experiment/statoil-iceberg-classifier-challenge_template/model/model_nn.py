import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 18 * 18, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def fit(X_train, y_train, X_valid, y_valid):
    # Convert data to PyTorch tensors and reshape it for convolutional layers
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Change to (batch_size, channels, height, width)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long).to(device)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    # Initialize the model, loss function and optimizer
    model = CNNModel(input_channels=3, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Train the model
    num_epochs = 50
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
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}")

    return model

def predict(model, X):
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = F.softmax(outputs, dim=1)
    return probabilities.cpu().numpy()[:, 1]