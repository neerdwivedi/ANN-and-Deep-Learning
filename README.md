# ANN and Deep Learning

This repository contains my lab work for the **Artificial Neural Networks and Deep Learning** course. Each lab is kept in its own folder with the Python code and a README explaining the concept and the code step by step.

## Repository Structure

```
ANN and Deep Learning/
├── README.md          <- this file (overview of all labs)
├── Lab 1/
│   ├── perceptron.py  <- perceptron built from scratch using NumPy
│   └── README.md      <- explanation of Lab 1
└── Lab 2/
    ├── and_gate_nn.py <- neural network for the AND gate using TensorFlow
    └── README.md      <- explanation of Lab 2
```

## Labs

### Lab 1 — Perceptron using NumPy

A perceptron (the smallest type of neural network) built from scratch with NumPy. It learns the **AND gate** by adjusting its weights and bias whenever it makes a mistake. This lab shows the core idea of how a single neuron learns, using a step activation function.

Key concepts: weights, bias, step activation function, the perceptron learning rule.

### Lab 2 — AND Gate using TensorFlow

The same AND gate problem, this time solved with a small neural network built in **TensorFlow (Keras)**. It uses a hidden layer with ReLU activation and an output layer with sigmoid activation, and trains using the Adam optimizer. This lab shows how the same task is done with a real deep learning library instead of coding everything by hand.

Key concepts: TensorFlow/Keras, hidden layers, ReLU and sigmoid activation, loss functions, epochs.

## How to Run

Each lab is a standalone Python script.

```
# Lab 1 (needs NumPy)
pip install numpy
python "Lab 1/perceptron.py"

# Lab 2 (needs TensorFlow)
pip install tensorflow
python "Lab 2/and_gate_nn.py"
```

## Tools Used

- **Python** — programming language
- **NumPy** — numerical computations (Lab 1)
- **TensorFlow / Keras** — building and training neural networks (Lab 2)

## Author

Neer Dwivedi
