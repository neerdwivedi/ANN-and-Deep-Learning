#Aim: Implement a feedforward neural network using TensorFlow or PyTorch.
# PyTorch Implementation
import torch
import torch.nn as nn
import torch.optim as optim

# Sample dataset: AND logic gate
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# Define a feedforward neural network
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(out))
        return out

# Initialize model, loss, and optimizer
model = FeedforwardNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict
print("\nPredictions:")
with torch.no_grad():
    predictions = model(X)
    for i, pred in enumerate(predictions):
        print(f"Input: {X[i].tolist()} => Predicted: {round(float(pred), 4)} => Class: {int(pred >= 0.5)}")
