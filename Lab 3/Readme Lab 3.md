# Lab 3 of Artificial Neural Networks and Deep Learning

**Aim:** Implement a feedforward neural network using PyTorch (for the AND gate).

## Concepts Used

1) **PyTorch:** A popular Python library for building and training neural networks. It handles the maths of learning (like gradients) automatically.

2) **Feedforward Neural Network:** A network where the data moves in one direction only — from input, through the hidden layer, to the output. There are no loops going backward.

3) **Layer (`nn.Linear`):** A fully connected layer that multiplies inputs by weights and adds a bias. `nn.Linear(2, 4)` means 2 inputs going into 4 neurons.

4) **Activation Functions:**
   - **ReLU:** keeps positive values as they are and turns negatives into 0. It adds non-linearity to the hidden layer.
   - **Sigmoid:** squashes any number into a value between 0 and 1, good for a yes/no output.

5) **Loss Function (`BCELoss`):** Binary Cross Entropy Loss. It measures how far the predictions are from the correct answers for a yes/no problem. Smaller loss = better.

6) **Optimizer (Adam):** The algorithm that adjusts the weights to reduce the loss during training.

7) **Epoch:** One full pass through all the training data. The model trains for 100 epochs here.

---

## About The Code

### Importing the libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

*In short: brings in PyTorch and its tools for layers and optimization.*

`torch` is the main PyTorch library. `torch.nn` holds the building blocks for networks (layers, activation functions, loss functions). `torch.optim` holds the optimizers used for training.

### Defining the dataset

```python
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
```

*In short: `X` holds the AND gate inputs, `y` holds the correct answers.*

`X` is the input data (all combinations of two switches). `y` is the correct AND output for each one — only [1,1] gives 1. A `tensor` is PyTorch's version of an array.

### Defining the feedforward neural network

```python
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
```

*In short: builds the network — 2 inputs → hidden layer of 4 (ReLU) → 1 output (sigmoid).*

- The class inherits from `nn.Module`, which is the base for all PyTorch models.
- `fc1 = nn.Linear(2, 4)`: the hidden layer, taking 2 inputs into 4 neurons.
- `relu`: the activation applied after the hidden layer.
- `fc2 = nn.Linear(4, 1)`: the output layer, turning 4 values into 1.
- `sigmoid`: the activation applied to the output to get a 0–1 value.
- The `forward` method describes how the data flows: input → fc1 → ReLU → fc2 → sigmoid → output.

### Setting up model, loss, and optimizer

```python
model = FeedforwardNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

*In short: creates the model, picks the error measure, and picks the optimizer.*

- `model`: builds the network defined above.
- `criterion = nn.BCELoss()`: the loss function for yes/no problems.
- `optimizer = optim.Adam(...)`: the Adam optimizer with a learning rate of 0.01, which updates the weights during training.

### Training the model

```python
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

*In short: repeats the learning loop 100 times to reduce the error.*

Each epoch does four things:
1. `outputs = model(X)`: the model makes predictions (forward pass).
2. `loss = criterion(outputs, y)`: measures how wrong the predictions are.
3. `optimizer.zero_grad()`: clears the old gradients from the last step.
4. `loss.backward()` then `optimizer.step()`: works out how to change the weights (backpropagation) and then updates them.

Repeating this 100 times slowly reduces the loss and the model learns the pattern.

### Making predictions

```python
print("\nPredictions:")
with torch.no_grad():
    predictions = model(X)
    for i, pred in enumerate(predictions):
        print(f"Input: {X[i].tolist()} => Predicted: {round(float(pred), 4)} => Class: {int(pred >= 0.5)}")
```

*In short: shows the trained model's guesses, turned into 0 or 1.*

`with torch.no_grad()` tells PyTorch we are only predicting, not training, so it skips gradient tracking (faster). For each input it prints the raw sigmoid output and the final class, using a threshold of **0.5** (0.5 or above becomes 1, otherwise 0).

---

## The Result

After training, the model learns the AND gate: the prediction is close to 1 only for the input [1, 1] and close to 0 for the other three inputs. Before training, the outputs would be random because the weights start off untrained.

---

## Short Summary

This lab builds a feedforward neural network in PyTorch with one hidden layer (ReLU) and one output (sigmoid). It trains for 100 epochs using the Adam optimizer and Binary Cross Entropy loss, learning the AND gate. Each epoch does a forward pass, measures the loss, and updates the weights through backpropagation.