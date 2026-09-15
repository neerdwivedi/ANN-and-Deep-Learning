# Lab #6 - Keras MLP for Regression

**Student:** Neer Dwivedi  
**Roll No.:** 24215217

---

## Objective

To implement a Multi-Layer Perceptron (MLP) using Keras/TensorFlow for a regression problem and study how different **activation functions** and **loss functions** affect model performance.

For this practical, the model predicts **Salary** using **Years of Experience**.

---

## Dataset

**Dataset:** `Salary_dataset.csv`

The dataset contains 30 records and 2 useful columns:

| Column | Purpose |
|---|---|
| `YearsExperience` | Input feature (X) |
| `Salary` | Target variable (y) |

Salary is a continuous numerical value, so this is a **regression problem**.

### Dataset Observation

- Total records: **30**
- Total useful columns: **2**
- Missing values: **0**
- Years of Experience: **1.2 to 10.6 years**
- Mean Years of Experience: **5.41 years**
- Salary: **37,732 to 122,392**
- Mean Salary: **76,004**
- Salary standard deviation: **27,414.43**

The scatter plot of Years of Experience vs Salary shows a clear positive relationship. In general, salary increases as years of experience increase.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. The CSV dataset was loaded using Pandas.
2. The unnecessary index column was removed.
3. Missing values were checked. No missing values were found.
4. `YearsExperience` was selected as the input feature.
5. `Salary` was selected as the target variable.
6. The dataset was divided into **80% training data and 20% testing data**.
7. `StandardScaler` was used to scale the input and target training data.

The split resulted in:

- **24 training samples**
- **6 testing samples**

Scaling helps the neural network train more smoothly. The scaler was fitted only on the training data to avoid data leakage.

The predicted salary values were converted back to the original salary scale before calculating the final evaluation metrics.

---

## MLP Architecture

A simple MLP was used for all experiments:

```text
Input (YearsExperience)
        |
        v
Dense Layer - 32 neurons
        |
        v
Dense Layer - 16 neurons
        |
        v
Output Layer - 1 neuron
        |
        v
Predicted Salary