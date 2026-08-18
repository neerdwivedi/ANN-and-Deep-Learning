"""
Lab #6 - Keras MLP for Regression
Student: Neer Dwivedi (24215217)

Predicts Salary from YearsExperience with a Multi-Layer Perceptron (Keras/TensorFlow),
comparing activation functions (ReLU, Sigmoid, Tanh) and loss functions (MSE, MAE, Huber).

Run in VS Code:
    pip install tensorflow scikit-learn pandas matplotlib
    python lab6_keras_mlp_regression.py
Keep Salary_dataset.csv in the same folder as this script.
Plots are shown on screen and also saved as PNG files next to the script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# 1. Load and explore the dataset
# ----------------------------------------------------------------------
def load_data():
    path = os.path.join(HERE, "Salary_dataset.csv")
    df = pd.read_csv(path, index_col=0)
    print("Shape:", df.shape)
    print(df.describe())
    print("\nMissing values per column:\n", df.isna().sum())
    return df


# ----------------------------------------------------------------------
# 2. Preprocess: scale X and y, split train/test
# ----------------------------------------------------------------------
def preprocess(df):
    X = df[["YearsExperience"]].values
    y = df[["Salary"]].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    data = {
        "X_train_s": x_scaler.transform(X_train),
        "X_test_s": x_scaler.transform(X_test),
        "y_train_s": y_scaler.transform(y_train),
        "y_test": y_test,
        "y_scaler": y_scaler,
    }
    print(f"\nTrain: {data['X_train_s'].shape}  Test: {data['X_test_s'].shape}")
    return data


# ----------------------------------------------------------------------
# 3. MLP: input -> Dense(32) -> Dense(16) -> linear output
# ----------------------------------------------------------------------
def build_model(activation, loss):
    model = keras.Sequential([
        keras.Input(shape=(1,)),
        layers.Dense(32, activation=activation),
        layers.Dense(16, activation=activation),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss=loss)
    return model


def train(activation, loss, label, data):
    tf.keras.utils.set_random_seed(42)
    model = build_model(activation, loss)
    history = model.fit(
        data["X_train_s"], data["y_train_s"],
        validation_split=0.2, epochs=200, batch_size=8, verbose=0,
    )
    pred = data["y_scaler"].inverse_transform(
        model.predict(data["X_test_s"], verbose=0)
    )
    mse = mean_squared_error(data["y_test"], pred)
    result = {
        "label": label,
        "activation": activation,
        "loss": loss if isinstance(loss, str) else "huber",
        "MAE": mean_absolute_error(data["y_test"], pred),
        "RMSE": float(np.sqrt(mse)),
        "MSE": float(mse),
        "R2": r2_score(data["y_test"], pred),
        "history": history.history,
    }
    print(f"{label:16s}  MAE={result['MAE']:9.1f}  "
          f"RMSE={result['RMSE']:9.1f}  R2={result['R2']:.4f}")
    return result


# ----------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------
def plot_loss_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results[:3]:  # activation experiment
        axes[0].plot(r["history"]["val_loss"], label=r["label"])
    axes[0].set_title("Experiment A - Activations (validation MSE, scaled)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Scaled loss")
    axes[0].legend(); axes[0].grid(alpha=.3)

    relu_mse = results[0]
    axes[1].plot(relu_mse["history"]["loss"], label="train")
    axes[1].plot(relu_mse["history"]["val_loss"], label="val")
    axes[1].set_title("ReLU + MSE - train vs validation loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Scaled loss")
    axes[1].legend(); axes[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "loss_curves_activations.png"), dpi=120)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, r in zip(axes, [results[0], results[3], results[4]]):
        ax.plot(r["history"]["loss"], label="train")
        ax.plot(r["history"]["val_loss"], label="val")
        ax.set_title(r["label"]); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "loss_curves_losses.png"), dpi=120)
    plt.show()


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main():
    print("TensorFlow", tf.__version__)
    df = load_data()
    data = preprocess(df)

    results = []
    # Experiment A: activation functions, loss fixed = MSE
    results.append(train("relu", "mse", "ReLU + MSE", data))
    results.append(train("sigmoid", "mse", "Sigmoid + MSE", data))
    results.append(train("tanh", "mse", "Tanh + MSE", data))
    # Experiment B: loss functions, activation fixed = ReLU
    results.append(train("relu", "mae", "ReLU + MAE", data))
    results.append(train("relu", keras.losses.Huber(delta=1.0), "ReLU + Huber", data))

    # Comparison table (the four models requested by the brief)
    table = pd.DataFrame([
        {"Experiment": "Model 1", "Activation": "ReLU", "Loss Function": "MSE",
         **{k: results[0][k] for k in ("MAE", "RMSE", "R2")}},
        {"Experiment": "Model 2", "Activation": "Sigmoid", "Loss Function": "MSE",
         **{k: results[1][k] for k in ("MAE", "RMSE", "R2")}},
        {"Experiment": "Model 3", "Activation": "Tanh", "Loss Function": "MSE",
         **{k: results[2][k] for k in ("MAE", "RMSE", "R2")}},
        {"Experiment": "Model 4", "Activation": "ReLU", "Loss Function": "Huber",
         **{k: results[4][k] for k in ("MAE", "RMSE", "R2")}},
    ])
    table["MAE"] = table["MAE"].round(1)
    table["RMSE"] = table["RMSE"].round(1)
    table["R2"] = table["R2"].round(4)
    print("\n=== Comparison table ===")
    print(table.to_string(index=False))
    table.to_csv(os.path.join(HERE, "comparison_table.csv"), index=False)

    best = max(results, key=lambda r: r["R2"])
    print(f"\nBest model by R2: {best['label']}  "
          f"(R2={best['R2']:.4f}, RMSE={best['RMSE']:.1f})")

    plot_loss_curves(results)
    print("\nSaved: loss_curves_activations.png, loss_curves_losses.png, "
          "comparison_table.csv")


if __name__ == "__main__":
    main()