
# Lab #7: Keras MLP for Multiclass Classification
# Dataset: Handwritten Digits (sklearn "load_digits", 10 classes, 0-9)
#
# This script builds a Multi-Layer Perceptron (MLP) with Keras/TensorFlow and:
#   - Experiment A: compares 3 hidden-layer activation functions (ReLU, Sigmoid, Tanh)
#   - Experiment B: compares 3 optimizers (Adam, SGD, RMSprop)
# It then plots learning curves, evaluates every model on the SAME test set,
# draws confusion matrices, and prints a comparison table.
#
# Written to be beginner-friendly: plain Keras Sequential API, lots of comments.
# Runs as-is in Google Colab (just paste into a cell) or locally with:
#   pip install tensorflow scikit-learn matplotlib seaborn


# ----------------------------- 0. Imports ------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Fix the random seed so results are the same every time we run (reproducibility).
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



# 1. DATASET PREPARATION

print("=" * 70)
print("1. DATASET PREPARATION")
print("=" * 70)

# Load the built-in handwritten digits dataset.
# Each sample is an 8x8 grayscale image flattened into 64 numbers (pixel values 0-16).
digits = load_digits()
X = digits.data        # shape (1797, 64)  -> 64 input features
y = digits.target      # shape (1797,)     -> label 0..9

n_features = X.shape[1]
n_classes = len(np.unique(y))

print(f"Total samples      : {X.shape[0]}")
print(f"Input features     : {n_features}  (8x8 pixel image flattened)")
print(f"Number of classes  : {n_classes}  (digits 0 through 9)")

# --- Class distribution: how many examples of each digit do we have? ---
unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution (digit : count):")
for u, c in zip(unique, counts):
    print(f"  {u} : {c}")
print("-> The classes are almost perfectly balanced (~180 each), so we do NOT")
print("   need any special class-imbalance handling.")

# --- Missing values check ---
print(f"\nMissing values in data: {np.isnan(X).sum()}  (0 means clean data)")

# --- Encode the target as one-hot vectors ---
# For softmax + categorical_crossentropy the label 3 becomes [0,0,0,1,0,0,0,0,0,0].
y_onehot = to_categorical(y, num_classes=n_classes)

# --- Train / test split (80% train, 20% test) ---
# stratify=y keeps the same class proportions in both train and test sets.
# This SAME split is reused by every model so the comparison is fair.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.20, random_state=SEED, stratify=y
)
# Keep the integer test labels too (handy for the confusion matrix / metrics).
y_test_int = np.argmax(y_test, axis=1)

# --- Feature scaling ---
# Neural networks train faster and more stably when inputs are on a similar scale.
# StandardScaler makes each feature have mean 0 and standard deviation 1.
# IMPORTANT: fit the scaler ONLY on training data, then apply to test data,
# so no information from the test set "leaks" into training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


=
# 2. MLP MODEL BUILDER

# A small helper that builds the SAME architecture every time and only changes
# the hidden-layer activation and the optimizer. Keeping everything else fixed
# is what makes Experiment A and Experiment B fair comparisons.
#
# Architecture:
#   Input layer      : 64 features
#   Hidden layer 1   : 64 neurons
#   Hidden layer 2   : 32 neurons
#   Output layer     : 10 neurons + softmax (one neuron per class)

def build_model(activation="relu", optimizer="adam"):
    model = Sequential([
        Input(shape=(n_features,)),          # input layer (64 features)
        Dense(64, activation=activation),    # hidden layer 1
        Dense(32, activation=activation),    # hidden layer 2
        Dense(n_classes, activation="softmax")  # output layer (10 classes)
    ])
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",  # standard loss for multiclass + softmax
        metrics=["accuracy"],
    )
    return model


# Shared training settings so nothing but the variable-under-test changes.
EPOCHS = 60
BATCH_SIZE = 32
VAL_SPLIT = 0.20   # 20% of the training data is held out each epoch for validation


def train_model(activation, optimizer, tag):
    """Build, train, and return (model, history) for one configuration."""
    print(f"\nTraining [{tag}] -> activation={activation}, optimizer={optimizer}")
    model = build_model(activation=activation, optimizer=optimizer)
    history = model.fit(
        X_train, y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,          # set to 1 if you want to watch every epoch
    )
    return model, history



# 3 & 4. RUN THE EXPERIMENTS

# Experiment A: fix optimizer = Adam, vary the activation function.
# Experiment B: fix activation = ReLU, vary the optimizer.
# Note: "ReLU + Adam" belongs to both experiments, so we train it once and reuse it.

print("\n" + "=" * 70)
print("3 & 4. TRAINING ALL MODELS")
print("=" * 70)

results = {}   # will hold everything we need for plots, tables and evaluation

# ---- Experiment A: activation functions (optimizer fixed = Adam) ----
experiment_a = {
    "ReLU + Adam":    ("relu",    "adam"),
    "Sigmoid + Adam": ("sigmoid", "adam"),
    "Tanh + Adam":    ("tanh",    "adam"),
}

# ---- Experiment B: optimizers (activation fixed = ReLU) ----
# We build fresh optimizer objects so each uses its own default learning rate.
experiment_b = {
    "ReLU + SGD":     ("relu", SGD()),
    "ReLU + RMSprop": ("relu", RMSprop()),
    # "ReLU + Adam" is already trained above, so we don't repeat it here.
}

all_configs = {**experiment_a, **experiment_b}

for tag, (act, opt) in all_configs.items():
    model, history = train_model(act, opt, tag)
    results[tag] = {"model": model, "history": history}



# 5. LEARNING CURVES (accuracy & loss vs epoch)

# These curves let us SEE overfitting (train keeps improving while validation
# stalls or gets worse), underfitting, and how fast each model converges.

def plot_curves(config_names, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in config_names:
        h = results[name]["history"].history
        axes[0].plot(h["val_accuracy"], label=name)
        axes[1].plot(h["val_loss"], label=name)
    axes[0].set_title(f"{title} - Validation Accuracy vs Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title(f"{title} - Validation Loss vs Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.show()


# One figure per model showing BOTH train and validation, to spot overfitting.
def plot_train_vs_val(name, filename):
    h = results[name]["history"].history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(h["accuracy"], label="Train")
    axes[0].plot(h["val_accuracy"], label="Validation")
    axes[0].set_title(f"{name} - Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(h["loss"], label="Train")
    axes[1].plot(h["val_loss"], label="Validation")
    axes[1].set_title(f"{name} - Loss"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.show()


print("\n" + "=" * 70)
print("5. PLOTTING LEARNING CURVES")
print("=" * 70)

plot_curves(list(experiment_a.keys()),
            "Experiment A (Activations)", "expA_activations.png")
plot_curves(["ReLU + Adam", "ReLU + SGD", "ReLU + RMSprop"],
            "Experiment B (Optimizers)", "expB_optimizers.png")
# Train-vs-validation for the best all-round model:
plot_train_vs_val("ReLU + Adam", "relu_adam_train_vs_val.png")


# 
# 6. MODEL EVALUATION (on the untouched test set)

print("\n" + "=" * 70)
print("6. EVALUATION ON TEST SET (metrics use macro-average)")
print("=" * 70)

for name, data in results.items():
    y_pred = np.argmax(data["model"].predict(X_test, verbose=0), axis=1)
    data["y_pred"] = y_pred
    data["accuracy"]  = accuracy_score(y_test_int, y_pred)
    data["precision"] = precision_score(y_test_int, y_pred, average="macro")
    data["recall"]    = recall_score(y_test_int, y_pred, average="macro")
    data["f1"]        = f1_score(y_test_int, y_pred, average="macro")


# --- Confusion matrix for the best model ---
def plot_confusion(name, filename):
    cm = confusion_matrix(y_test_int, results[name]["y_pred"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(n_classes), yticklabels=range(n_classes))
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.show()
    return cm


best_name = max(results, key=lambda k: results[k]["f1"])
cm_best = plot_confusion(best_name, "confusion_matrix_best.png")

print(f"\nDetailed classification report for best model ({best_name}):\n")
print(classification_report(y_test_int, results[best_name]["y_pred"], digits=3))



# 7. COMPARISON TABLE
print("=" * 70)
print("7. COMPARISON TABLE (all metrics measured on the SAME test set)")
print("=" * 70)
header = f"{'Model':<18}{'Activation':<11}{'Optimizer':<11}{'TestAcc':<9}{'Prec':<8}{'Recall':<8}{'F1':<8}"
print(header)
print("-" * len(header))

# activation/optimizer labels for printing
labels = {
    "ReLU + Adam":    ("ReLU", "Adam"),
    "Sigmoid + Adam": ("Sigmoid", "Adam"),
    "Tanh + Adam":    ("Tanh", "Adam"),
    "ReLU + SGD":     ("ReLU", "SGD"),
    "ReLU + RMSprop": ("ReLU", "RMSprop"),
}
order = ["ReLU + Adam", "Sigmoid + Adam", "Tanh + Adam", "ReLU + SGD", "ReLU + RMSprop"]
for name in order:
    d = results[name]
    act, opt = labels[name]
    print(f"{name:<18}{act:<11}{opt:<11}"
          f"{d['accuracy']:<9.3f}{d['precision']:<8.3f}{d['recall']:<8.3f}{d['f1']:<8.3f}")

# 9. FINAL MODEL SELECTION
print("\n" + "=" * 70)
print("9. FINAL MODEL SELECTION")
print("=" * 70)
print(f"Best model by macro F1-score: {best_name}")
print(f"  Test Accuracy : {results[best_name]['accuracy']:.3f}")
print(f"  Precision     : {results[best_name]['precision']:.3f}")
print(f"  Recall        : {results[best_name]['recall']:.3f}")
print(f"  F1-score      : {results[best_name]['f1']:.3f}")
print("\nSee the markdown report for the full written analysis and justification.")