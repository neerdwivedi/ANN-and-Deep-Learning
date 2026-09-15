
# Lab #7: Keras MLP for Multiclass Classification
# Dataset: Handwritten Digits (sklearn "load_digits", 10 classes, 0-9)
#
# This script builds a Multi-Layer Perceptron (MLP) with Keras/TensorFlow and:
#   - Experiment A: compares 3 hidden-layer activation functions (ReLU, Sigmoid, Tanh)
#   - Experiment B: compares 3 optimizers (Adam, SGD, RMSprop)
# It then plots learning curves, evaluates every model on the SAME test set,
# draws confusion matrices, and prints a comparison table.
#


"""
Lab 7 - Keras MLP for Multiclass Classification

Dataset: Handwritten Digits

Aim:
To build an MLP using TensorFlow/Keras for multiclass classification
and compare different activation functions and optimizers.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# -------------------------------------------------------
# BASIC SETTINGS
# -------------------------------------------------------

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 30
BATCH_SIZE = 32


# -------------------------------------------------------
# 1. LOAD THE DATASET
# -------------------------------------------------------

print("=" * 60)
print("1. DATASET")
print("=" * 60)

digits = load_digits()

X = digits.data
y = digits.target

n_features = X.shape[1]
n_classes = len(np.unique(y))

print("Total samples:", X.shape[0])
print("Input features:", n_features)
print("Number of classes:", n_classes)


# -------------------------------------------------------
# 2. CLASS DISTRIBUTION
# -------------------------------------------------------

print("\nClass Distribution:")

unique_classes, counts = np.unique(
    y,
    return_counts=True
)

for digit, count in zip(
    unique_classes,
    counts
):
    print(f"Digit {digit}: {count}")


# -------------------------------------------------------
# 3. CHECK MISSING VALUES
# -------------------------------------------------------

missing_values = np.isnan(X).sum()

print("\nMissing values:", missing_values)


# -------------------------------------------------------
# 4. SPLIT DATA
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -------------------------------------------------------
# 5. FEATURE SCALING
# -------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# -------------------------------------------------------
# 6. MODEL BUILDING FUNCTION
# -------------------------------------------------------

def build_model(activation, optimizer):

    model = keras.Sequential([

        # Input layer - 64 pixel features
        layers.Input(
            shape=(n_features,)
        ),

        # Hidden layer 1
        layers.Dense(
            64,
            activation=activation
        ),

        # Hidden layer 2
        layers.Dense(
            32,
            activation=activation
        ),

        # Output layer - 10 classes
        layers.Dense(
            n_classes,
            activation="softmax"
        )

    ])

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------------------------------
# 7. FUNCTION TO TRAIN A MODEL
# -------------------------------------------------------

def train_model(name, activation, optimizer):

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model = build_model(
        activation,
        optimizer
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    print(
        f"Final Training Accuracy: "
        f"{history.history['accuracy'][-1] * 100:.2f}%"
    )

    print(
        f"Final Validation Accuracy: "
        f"{history.history['val_accuracy'][-1] * 100:.2f}%"
    )

    print(
        f"Final Training Loss: "
        f"{history.history['loss'][-1]:.4f}"
    )

    print(
        f"Final Validation Loss: "
        f"{history.history['val_loss'][-1]:.4f}"
    )

    return model, history


# -------------------------------------------------------
# 8. EXPERIMENT A - ACTIVATION FUNCTIONS
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("EXPERIMENT A - ACTIVATION FUNCTIONS")
print("Optimizer is fixed as Adam")
print("=" * 60)


models = {}
histories = {}


# ReLU + Adam
models["ReLU + Adam"], histories["ReLU + Adam"] = train_model(
    "ReLU + Adam",
    "relu",
    "adam"
)


# Sigmoid + Adam
models["Sigmoid + Adam"], histories["Sigmoid + Adam"] = train_model(
    "Sigmoid + Adam",
    "sigmoid",
    "adam"
)


# Tanh + Adam
models["Tanh + Adam"], histories["Tanh + Adam"] = train_model(
    "Tanh + Adam",
    "tanh",
    "adam"
)


# -------------------------------------------------------
# 9. EXPERIMENT B - OPTIMIZERS
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("EXPERIMENT B - OPTIMIZERS")
print("Activation is fixed as ReLU")
print("=" * 60)


# ReLU + SGD
models["ReLU + SGD"], histories["ReLU + SGD"] = train_model(
    "ReLU + SGD",
    "relu",
    keras.optimizers.SGD()
)


# ReLU + RMSprop
models["ReLU + RMSprop"], histories["ReLU + RMSprop"] = train_model(
    "ReLU + RMSprop",
    "relu",
    keras.optimizers.RMSprop()
)


# ReLU + Adam was already trained above,
# so we do not need to train it again.


# -------------------------------------------------------
# 10. EVALUATE ALL MODELS
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


results = {}


for name, model in models.items():

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions
    }


    print(f"\n{name}")

    print(
        f"Test Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision     : {precision:.4f}"
    )

    print(
        f"Recall        : {recall:.4f}"
    )

    print(
        f"F1-Score      : {f1:.4f}"
    )


# -------------------------------------------------------
# 11. COMPARISON TABLE
# -------------------------------------------------------

print("\n")
print("=" * 80)
print("COMPARISON TABLE")
print("=" * 80)


print(
    f"{'Model':<20}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1-Score':<12}"
)

print("-" * 80)


for name, result in results.items():

    print(
        f"{name:<20}"
        f"{result['accuracy']:<12.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['f1']:<12.4f}"
    )


# -------------------------------------------------------
# 12. FIND THE BEST MODEL
# -------------------------------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)


print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Best Model:",
    best_model_name
)

print(
    f"Test Accuracy: "
    f"{results[best_model_name]['accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{results[best_model_name]['precision']:.4f}"
)

print(
    f"Recall: "
    f"{results[best_model_name]['recall']:.4f}"
)

print(
    f"F1-Score: "
    f"{results[best_model_name]['f1']:.4f}"
)


# -------------------------------------------------------
# 13. CONFUSION MATRIX OF BEST MODEL
# -------------------------------------------------------

cm = confusion_matrix(
    y_test,
    results[best_model_name]["predictions"]
)


print("\nConfusion Matrix:")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(n_classes)
)

display.plot()

plt.title(
    "Confusion Matrix - " + best_model_name
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.close()


print(
    "Confusion matrix saved as confusion_matrix.png"
)


# -------------------------------------------------------
# 14. ACTIVATION FUNCTION COMPARISON GRAPH
# -------------------------------------------------------

activation_models = [
    "ReLU + Adam",
    "Sigmoid + Adam",
    "Tanh + Adam"
]


plt.figure(
    figsize=(8, 5)
)


for name in activation_models:

    plt.plot(
        histories[name].history["val_accuracy"],
        label=name
    )


plt.xlabel("Epoch")

plt.ylabel("Validation Accuracy")

plt.title(
    "Activation Function Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "activation_comparison.png"
)

plt.close()


print(
    "Activation comparison saved as activation_comparison.png"
)


# -------------------------------------------------------
# 15. OPTIMIZER COMPARISON GRAPH
# -------------------------------------------------------

optimizer_models = [
    "ReLU + Adam",
    "ReLU + SGD",
    "ReLU + RMSprop"
]


plt.figure(
    figsize=(8, 5)
)


for name in optimizer_models:

    plt.plot(
        histories[name].history["val_accuracy"],
        label=name
    )


plt.xlabel("Epoch")

plt.ylabel("Validation Accuracy")

plt.title(
    "Optimizer Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "optimizer_comparison.png"
)

plt.close()


print(
    "Optimizer comparison saved as optimizer_comparison.png"
)


# -------------------------------------------------------
# 16. TRAINING VS VALIDATION CURVES
#     FOR THE BEST MODEL
# -------------------------------------------------------

best_history = histories[
    best_model_name
]


epochs_range = range(
    1,
    EPOCHS + 1
)


# Accuracy graph
plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    best_history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    best_history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy - "
    + best_model_name
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "best_model_accuracy.png"
)

plt.close()


# Loss graph
plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    best_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    best_history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss - "
    + best_model_name
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "best_model_loss.png"
)

plt.close()


print(
    "Best model accuracy and loss graphs saved."
)


# -------------------------------------------------------
# 17. FINAL OUTPUT
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("FILES GENERATED")
print("=" * 60)

print("1. confusion_matrix.png")
print("2. activation_comparison.png")
print("3. optimizer_comparison.png")
print("4. best_model_accuracy.png")
print("5. best_model_loss.png")

print("\nLab 7 completed successfully.")