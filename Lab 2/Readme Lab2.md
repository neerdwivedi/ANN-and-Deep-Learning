# Lab 2 of Artificial Neural Networks and Deep Learning

**Aim:** Build a simple neural network using TensorFlow (Keras) to learn the AND gate.

## Concepts Used

1) **TensorFlow:** A Python library used to build and train neural networks. It handles all the heavy maths of learning automatically.

2) **Keras:** A simple, high-level part of TensorFlow used to build models by stacking layers together.

3) **Neural Network:** A group of connected neurons arranged in layers. Unlike a single perceptron, it can have hidden layers, which let it learn more complex patterns.

4) **Hidden Layer:** A layer between the input and the output. It processes the inputs and passes the result forward. Here it has 4 neurons.

5) **Activation Functions:**
   - **ReLU** (used in the hidden layer): keeps positive values as they are and turns negative values into 0. It adds non-linearity so the network can learn complex patterns.
   - **Sigmoid** (used in the output layer): squashes any number into a value between 0 and 1, which is perfect for a yes/no (binary) answer.

6) **Epoch:** One full pass through all the training examples. Training for 10 epochs means the network sees the whole dataset 10 times.

---

## About The Code

### Importing the libraries

```python
import tensorflow as tf
import numpy as np
```

*In short: brings in TensorFlow for the neural network and NumPy for the numbers.*

`tensorflow` is used to build and train the neural network. `numpy` is used for numerical computations and handling arrays.

### Defining the dataset

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
```

*In short: `X` holds the AND gate inputs, `y` holds the correct answers.*

`X` is the input dataset representing the AND gate inputs. `y` is the output label, where the output is 1 only when both inputs are 1, following the truth table of the AND gate.

### Building the neural network model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=2, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')             # Output layer
])
```

*In short: stacks a hidden layer of 4 neurons and an output layer of 1 neuron.*

- **Sequential:** Creates a straight stack of layers, one after another.
- **First layer (hidden layer):**
  - `Dense(4)`: a fully connected layer with 4 neurons.
  - `input_dim=2`: accepts 2 input values (each sample has two numbers).
  - `activation='relu'`: uses ReLU to add non-linearity.
- **Second layer (output layer):**
  - `Dense(1)`: gives a single output value.
  - `activation='sigmoid'`: outputs a value between 0 and 1, suitable for binary classification.

### Compiling the model

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

*In short: sets how the model will learn and how its progress is measured.*

- `optimizer='adam'`: an efficient algorithm that adjusts the weights during training.
- `loss='binary_crossentropy'`: the error measure used for yes/no (binary) problems.
- `metrics=['accuracy']`: tracks how accurate the model is while training.

### Predictions before training

```python
print("Predictions BEFORE training:")
untrained_predictions = model.predict(X)
for i, pred in enumerate(untrained_predictions):
    print(f"Input: {X[i]} => Predicted: {round(float(pred), 4)} => Class: {int(pred >= 0)}")
```

*In short: shows the model's guesses before it has learned anything.*

At this point the model has random weights, so its answers are basically random. This is printed to compare against the results after training.

### Training the model

```python
model.fit(X, y, epochs=10, verbose=2)
```

*In short: teaches the model on the data for 10 rounds.*

`fit` trains the model on the inputs `X` and answers `y` for 10 epochs. During each epoch the model adjusts its weights to reduce mistakes. `verbose=2` controls how much detail is printed during training.

### Evaluating and making predictions

```python
predictions = model.predict(X)
for i, pred in enumerate(predictions):
    print(f"Input: {X[i]} => Predicted: {round(float(pred), 4)} => Class: {int(pred >= 0.5)}")
```

*In short: shows the model's guesses after training and turns them into 0 or 1.*

The trained model predicts an output for every input combination. The raw output is a value between 0 and 1 (from sigmoid). A threshold of **0.5** is used to convert it into a class label: 0.5 or above becomes 1, otherwise 0.

---

## The Result

Before training, the model behaves randomly because it has no knowledge of the pattern. After training, it learns to approximate the AND function, giving values close to 1 only for the input [1, 1] and values close to 0 for the rest.

---

## Short Summary

This lab uses TensorFlow to build a small neural network with one hidden layer to learn the AND gate. The model starts with random weights, trains over several epochs using the Adam optimizer, and gradually learns the pattern. The sigmoid output is turned into a 0 or 1 answer using a 0.5 threshold.