Lab 1 of Artificial Neurak Netwroks and Deep Learning
Aim : Write a simple python program to simulate a perceptron using Numpy

1) Numpy : Numpy is an inbuilt python library. It is imported as "import numppy as np".
2) Perceptron: Is the smallest type of neural network. It acts like a binary classifier. It takes an input, uses an activation function and gives an output.
3) Activation Function : An activation function decides how much of a neuron's information should be passed to the next layer. determines whether a neuron fires by applying a mathematical transformation to its weighted inputs. It is crucial because it introduces non-linearity into the network, enabling the model to learn and represent complex patterns like speech recognition and image classification. Without it, the network acts merely as a linear regression model.


## About The Code

### Importing the toolkit

```python
import numpy as np
```

This brings in a maths toolkit called NumPy so the calculations can be done quickly. `np` is just a short nickname for it.

### The decision rule

```python
def step_function(x):
    return 1 if x >= 0 else 0
```

This is the  decision rule. It takes a number. If the number is 0 or bigger, the answer is **1** (yes). If the number is negative, the answer is **0** (no).

### Building the perceptron

```python
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate
```

A class is a blueprint for building the perceptron. When the perceptron is built, this setup runs:

- **weights** — how much importance the perceptron gives to each input. They all start at 0 because the perceptron knows nothing yet.
- **bias** — a small nudge that makes it easier or harder for the perceptron to say "yes." It also starts at 0.
- **lr (learning rate)** — how big a step the perceptron takes when it fixes a mistake. A small value like 0.1 means careful, gentle changes.

### Making a guess

```python
def predict(self, x):
    z = np.dot(x, self.weights) + self.bias
    return step_function(z)
```

This is the perceptron making a guess:
> np.dot calculates the dot product of two sets of numbers , meaning it multiplies matching pairs together and adds up all the results into a single number

1. Multiply each input by its weight and add them all together (`np.dot` does the multiply-and-add in one step).
2. Add the bias.
3. Send that total into the step function, which gives back a 0 or a 1.

Example: inputs [1, 1], weights [0.2, 0.2], bias -0.3:
(1 × 0.2) + (1 × 0.2) - 0.3 = 0.1. Since 0.1 is 0 or bigger, the answer is 1.

### Learning from practice

```python
def train(self, X, y, epochs=10):
    for epoch in range(epochs):
        for xi, target in zip(X, y):
            prediction = self.predict(xi)
            error = target - prediction
            self.weights += self.lr * error * xi
            self.bias += self.lr * error
```

This is the perceptron practicing and learning. It is the most important part:

- **epochs = 10** means the perceptron goes through all the practice examples 10 times. Practice makes perfect.
- For each example, `xi` is the input and `target` is the correct answer that is already known.
- **Guess:** the perceptron makes a prediction.
- **Check:** error = target - prediction.
  - Correct answer: error is 0, so nothing changes.
  - perceptron said 0 but should have said 1: error is +1, so the weights are pushed up.
  - perceptron said 1 but should have said 0: error is -1, so the weights are pushed down.
- **Fix:** the weights and bias change a little in the right direction.

By repeating this again and again, the mistakes get smaller until the perceptron gets everything right.

### The practice material

```python
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
```

`X` holds every combination of the two inputs. `y` holds the correct AND answer for each one. Only the last pair, [1,1], gives an answer of 1.

### Training the perceptron

```python
p = Perceptron(input_size=2)
p.train(X, y, epochs=10)
```

This builds a perceptron with 2 inputs (because there are two switches) and then lets it practice for 10 rounds.

### Testing the perceptron

```python
print("Predictions:")
for xi in X:
    print(f"Input: {xi}, Output: {p.predict(xi)}")
```

This is the exam. Each input is fed back in, and the  answer is printed.

---

## The Result

```
Input: [0 0], Output: 0
Input: [0 1], Output: 0
Input: [1 0], Output: 0
Input: [1 1], Output: 1
```

Only [1 1] gives 1. The perceptron learned the AND rule all on its own.

---

## Short Summary

The perceptron starts with no knowledge. It keeps guessing on examples, and every time it is wrong it changes its importance numbers a little. This continues until the perceptron stops making mistakes and has learned the rule.