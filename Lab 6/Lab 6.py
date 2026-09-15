# ============================================================
# LAB 6: Keras MLP for Regression
# Dataset: Salary Dataset
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Set random seed for reproducible results
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Get the folder where this Python file is located
current_folder = os.path.dirname(os.path.abspath(__file__))

# Dataset is inside the "dataset" folder
dataset_path = os.path.join(
    current_folder,
    "dataset",
    "Salary_dataset.csv"
)

df = pd.read_csv(dataset_path)

# Remove unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())


# ============================================================
# 3. VISUALIZE THE DATA
# ============================================================

plt.scatter(df["YearsExperience"], df["Salary"])

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Years of Experience vs Salary")

plt.show()


# ============================================================
# 4. DEFINE INPUT AND TARGET
# ============================================================

# Input feature
X = df[["YearsExperience"]].values

# Target variable
y = df[["Salary"]].values


# ============================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. SCALE THE DATA
# ============================================================

x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Fit scaler only on training data
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)


# ============================================================
# 7. CREATE THE MLP MODEL
# ============================================================

def create_model(activation="relu", loss="mse"):

    model = keras.Sequential([
        # Input layer
        keras.Input(shape=(1,)),

        # First hidden layer
        layers.Dense(32, activation=activation),

        # Second hidden layer
        layers.Dense(16, activation=activation),

        # Output layer
        # Linear activation is used for regression
        layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=loss
    )

    return model


# ============================================================
# 8. FUNCTION TO TRAIN AND EVALUATE MODEL
# ============================================================

def train_model(activation, loss):

    # Reset random seed so comparisons are fair
    tf.keras.utils.set_random_seed(42)

    model = create_model(
        activation=activation,
        loss=loss
    )

    history = model.fit(
        X_train_scaled,
        y_train_scaled,

        # 20% of training data is used for validation
        validation_split=0.2,

        epochs=200,
        batch_size=8,
        verbose=0
    )

    # Predict scaled salary values
    y_pred_scaled = model.predict(
        X_test_scaled,
        verbose=0
    )

    # Convert predictions back to actual salary values
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    return model, history, mae, rmse, r2


# ============================================================
# 9. EXPERIMENT 1: ACTIVATION FUNCTIONS
# ============================================================
#
# We change the activation function while keeping
# the loss function fixed as MSE.
#
# ReLU vs Sigmoid vs Tanh
# ============================================================

activations = [
    "relu",
    "sigmoid",
    "tanh"
]

activation_results = []

activation_histories = {}


for activation in activations:

    model, history, mae, rmse, r2 = train_model(
        activation=activation,
        loss="mse"
    )

    activation_histories[activation] = history

    activation_results.append([
        activation,
        "MSE",
        mae,
        rmse,
        r2
    ])


# ============================================================
# 10. ACTIVATION FUNCTION COMPARISON TABLE
# ============================================================

activation_table = pd.DataFrame(
    activation_results,
    columns=[
        "Activation",
        "Loss",
        "MAE",
        "RMSE",
        "R2"
    ]
)

print("\n======================================")
print("ACTIVATION FUNCTION COMPARISON")
print("======================================")

print(activation_table.round(4))


# ============================================================
# 11. PLOT ACTIVATION FUNCTION LOSS CURVES
# ============================================================

plt.figure(figsize=(8, 5))


for activation in activations:

    plt.plot(
        activation_histories[activation].history["val_loss"],
        label=activation
    )


plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Activation Function Comparison")

plt.legend()

plt.show()


# ============================================================
# 12. EXPERIMENT 2: LOSS FUNCTIONS
# ============================================================
#
# Here activation is kept fixed as ReLU.
#
# MSE vs MAE vs Huber
# ============================================================

losses = {
    "MSE": "mse",
    "MAE": "mae",
    "Huber": keras.losses.Huber()
}


loss_results = []

loss_histories = {}


for loss_name, loss_function in losses.items():

    model, history, mae, rmse, r2 = train_model(
        activation="relu",
        loss=loss_function
    )

    loss_histories[loss_name] = history

    loss_results.append([
        "ReLU",
        loss_name,
        mae,
        rmse,
        r2
    ])


# ============================================================
# 13. LOSS FUNCTION COMPARISON TABLE
# ============================================================

loss_table = pd.DataFrame(
    loss_results,
    columns=[
        "Activation",
        "Loss",
        "MAE",
        "RMSE",
        "R2"
    ]
)


print("\n======================================")
print("LOSS FUNCTION COMPARISON")
print("======================================")

print(loss_table.round(4))


# ============================================================
# 14. PLOT TRAINING AND VALIDATION LOSS
# ============================================================

for loss_name in losses:

    plt.figure(figsize=(7, 4))

    # Training loss
    plt.plot(
        loss_histories[loss_name].history["loss"],
        label="Training Loss"
    )

    # Validation loss
    plt.plot(
        loss_histories[loss_name].history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "ReLU + " + loss_name
    )

    plt.legend()

    plt.show()


# ============================================================
# 15. FINAL COMPARISON TABLE
# ============================================================

# Combine activation and loss experiments
final_table = pd.concat(
    [
        activation_table,
        loss_table
    ],
    ignore_index=True
)


# ReLU + MSE exists in both experiments.
# Remove the duplicate.
final_table = final_table.drop_duplicates(
    subset=["Activation", "Loss"]
)


print("\n======================================")
print("FINAL MODEL COMPARISON")
print("======================================")

print(final_table.round(4))


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

# Higher R2 indicates better regression performance
best_index = final_table["R2"].idxmax()

best_model = final_table.loc[best_index]


print("\n======================================")
print("BEST MODEL")
print("======================================")

print(best_model)