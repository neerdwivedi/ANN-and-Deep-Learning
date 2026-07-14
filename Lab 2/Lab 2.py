## 1. Importing Libraries
import tensorflow as tf
import numpy as np
#tensorflow: Used to build and train neural networks.
#numpy: Used for numerical computations and array manipulations.

# 2. Defining the Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
# X is the input dataset representing the AND gate inputs.
# y is the output label, where the output is 1 only when both inputs are 1, as per the truth table of the AND gate.#

#3. Building the Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=2, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')              # Output layer
])
# Sequential: Creates a linear stack of layers.
# First Layer (Hidden Layer):
# Dense(4): Fully connected layer with 4 neurons.
# input_dim=2: Accepts 2 input features (each input sample has two values).
# activation='relu': Uses ReLU activation to introduce non-linearity.
# Second Layer (Output Layer):
# Dense(1): Outputs a single value.
# activation='sigmoid': Outputs a value between 0 and 1, suitable for binary classification.#

# 4. Compiling the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# optimizer='adam': Efficient optimization algorithm.
# loss='binary_crossentropy': Used for binary classification problems.
# metrics=['accuracy']: Tracks accuracy during training.

#### Predictions BEFORE training
print("Predictions BEFORE training:")
untrained_predictions = model.predict(X)
for i, pred in enumerate(untrained_predictions):
    print(f"Input: {X[i]} => Predicted: {round(float(pred), 4)} => Class: {int(pred >= 0)}")

# 5. Training the Model
model.fit(X, y, epochs=10, verbose=2)
# Trains the model for 10 epochs.
#verbose=0: Suppresses output during training.

# 6. Evaluating and Making Predictions
predictions = model.predict(X)
#Predicts outputs for all input combinations.
for i, pred in enumerate(predictions):
    print(f"Input: {X[i]} => Predicted: {round(float(pred), 4)} => Class: {int(pred >= 0.5)}")

# Rounds and prints the predicted output and its classified label (0 or 1).
#A threshold of 0.5 is used to convert the sigmoid output to class labels.

# Result
# Before training, the model behaves randomly because it has no knowledge of the pattern.
# After training, it learns to approximate the AND function.