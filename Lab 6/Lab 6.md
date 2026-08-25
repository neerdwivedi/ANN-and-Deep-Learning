# Lab #6 — Keras MLP for Regression


---

## Objective

Implement a Multi-Layer Perceptron (MLP) using Keras/TensorFlow for a regression problem, and analyse how different **activation functions** (ReLU, Sigmoid, Tanh) and **loss functions** (MSE, MAE, Huber) affect model performance. The emphasis is not merely on swapping functions and reporting numbers, but on **justifying** each choice against the characteristics of the dataset and the regression objective.

---

## 1. Dataset Description and Source

The dataset used is `Salary_dataset.csv`, the classic single-feature *Salary Data* set commonly hosted on Kaggle. It contains 30 records of employees with the following columns:

| Column | Type | Role |
|---|---|---|
| `YearsExperience` | Continuous numeric | Input feature (X) |
| `Salary` | Continuous numeric | Target variable (y) |

Because the target `Salary` is a continuous quantity, this is a **regression** problem. There is a single predictor (`YearsExperience`) whose relationship with salary is monotonic and approximately linear. Summary statistics from the data:

- 30 rows, 2 columns after dropping the index.
- `YearsExperience` ranges from 1.2 to 10.6 years (mean ≈ 5.41).
- `Salary` ranges from 37,732 to 122,392 (mean ≈ 76,004, standard deviation ≈ 27,414).

This small, clean, one-feature dataset is an ideal benchmark for isolating the effect of activation and loss choices, because the underlying signal is simple enough that differences in model behaviour can be attributed to those design decisions rather than to noise in a complex feature space.

---

## 2. Data Exploration

A scatter plot of `Salary` against `YearsExperience` shows a clear positive, near-linear trend: salary rises steadily with experience, with mild scatter around the trend line. There are no obvious extreme outliers. This visual confirms that a regression model should be able to fit the relationship well, and that the target is unbounded and increasing — a fact that guides both the output-layer and loss-function choices below.

---

## 3. Data Preprocessing

**Missing values.** A null check (`df.isna().sum()`) returns zero missing values in both columns, so no imputation is required.

**Categorical variables.** The only feature is numeric (`YearsExperience`), so there are no categorical variables to encode. No one-hot or label encoding is needed.

**Feature scaling / normalisation.** Both the input `X` and the target `y` are standardised with `StandardScaler` (zero mean, unit variance), fit on the training set only to avoid data leakage. Two reasons:

- Scaling the **input** keeps the gradients well-conditioned and lets the Adam optimizer converge smoothly.
- Scaling the **target** is important in this lab specifically because two of the activations tested — **Sigmoid** and **Tanh** — are bounded and saturate for large inputs. Centring the target near zero keeps the network operating in the responsive (non-saturated) region of those functions, giving them a fair chance to compete with ReLU.

All evaluation metrics are computed back on the **original salary scale** by inverse-transforming the model predictions, so the reported MAE and RMSE are in real currency units and are directly interpretable.

**Train/test split.** The data is split 80% train / 20% test using a fixed random seed (`random_state=42`). Every experiment uses this identical split so that differences between models come only from the activation/loss change and not from a different sample of data. This leaves 24 training rows and 6 test rows.

---

## 4. MLP Model Development

A single compact architecture is used and kept **identical across every experiment** so that only the activation function or loss function varies:

| Layer | Units | Activation |
|---|---|---|
| Input | 1 | — |
| Hidden layer 1 | 32 | *varies (ReLU / Sigmoid / Tanh)* |
| Hidden layer 2 | 16 | *varies (ReLU / Sigmoid / Tanh)* |
| Output | 1 | Linear |

This satisfies the requirement of an input layer, **at least two hidden layers**, and a single output neuron for regression. The output activation is deliberately **linear** because the salary target is continuous and unbounded — a bounded output (e.g. sigmoid) could not represent arbitrarily large salaries.

**Training configuration (held constant):**

- Optimizer: **Adam**, learning rate 0.01
- Batch size: **8**
- Epochs: **200**
- Validation: 20% of the training set held out (`validation_split=0.2`) to monitor generalisation during training
- Fixed random seed for reproducibility

---

## 5. Experiment A — Activation Functions (loss fixed = MSE)

Three activation functions are compared while the loss is held at MSE. The justification for each, tied to this dataset:

- **ReLU** — The salary–experience relationship rises monotonically. ReLU's non-saturating positive region models a steadily increasing target efficiently and trains quickly, and it does not suffer from vanishing gradients.
- **Sigmoid** — Bounded to (0, 1) and saturating. It is included to demonstrate how a saturating unit behaves; because the target is standardised, its useful range is respected and it remains competitive on this simple task, though on deeper networks its saturation would slow learning.
- **Tanh** — Zero-centred and bounded to (−1, 1). It typically converges faster than Sigmoid and pairs naturally with a standardised, zero-mean target.

---

## 6. Experiment B — Loss Functions (activation fixed = ReLU)

Three regression loss functions are compared while the activation is held at ReLU. The justification for each, tied to the salary-regression objective:

- **Mean Squared Error (MSE)** — Squares the residuals, so it penalises large salary misses most heavily and implicitly assumes roughly Gaussian noise. It is the natural default for this clean dataset, but it is sensitive to outliers.
- **Mean Absolute Error (MAE)** — Penalises errors linearly, which makes it **robust to outliers**: a few unusually high or low salaries cannot dominate the training signal. It directly optimises the median-style absolute error we report.
- **Huber Loss** — Behaves quadratically for small errors and linearly for large ones. It keeps MSE's smooth, well-behaved gradients near the optimum while gaining MAE's robustness to the occasional extreme salary, making it a strong compromise for real-world salary data.

---

## 7. Model Training and Loss Curves

Each model was trained on the same split for 200 epochs, recording both training and validation loss per epoch. Two sets of learning curves were produced:

- **Experiment A curves** — validation loss (scaled MSE) for ReLU, Sigmoid and Tanh on a shared axis, plus a train-vs-validation plot for the ReLU + MSE baseline.
- **Experiment B curves** — train-vs-validation loss for ReLU with MSE, MAE and Huber, each on its own axis (absolute loss values are not comparable across different loss functions, so the shape of convergence is compared rather than the raw magnitude).

The curves show all activations driving the validation loss down and then flattening, with ReLU reaching its low point fastest and Sigmoid/Tanh close behind. None of the validation curves rise sharply after their minimum, which argues against severe overfitting despite the very small dataset.

---

## 8. Model Evaluation and Comparison Table

Models were evaluated on the held-out test set using MAE, RMSE and R². The four models requested by the brief are the three activations at fixed MSE plus ReLU retrained with the robust Huber loss.

| Experiment | Activation | Loss Function | MAE | RMSE | R² |
|---|---|---|---|---|---|
| Model 1 | ReLU | MSE | 6839.3 | 8161.9 | 0.8696 |
| Model 2 | Sigmoid | MSE | 6004.7 | 7579.0 | 0.8875 |
| Model 3 | Tanh | MSE | 6002.3 | 7598.4 | 0.8870 |
| Model 4 | ReLU | Huber | 7081.3 | 8439.1 | 0.8606 |

Additional run from Experiment B (ReLU + MAE), for completeness:

| Model | Activation | Loss Function | MAE | RMSE | R² |
|---|---|---|---|---|---|
| — | ReLU | MAE | 7074.3 | 8507.8 | 0.8583 |

**Best model by R²:** Sigmoid + MSE (R² = 0.8875, RMSE ≈ 7579), with Tanh + MSE essentially tied.

---

## 9. Comparison, Interpretation and Analysis

**Which activation function performed best?**
On the held-out test set the three activations were close, with the two bounded, target-matched activations slightly ahead: **Sigmoid + MSE (R² ≈ 0.888)** and **Tanh + MSE (R² ≈ 0.887)** narrowly outperformed **ReLU + MSE (R² ≈ 0.870)**. Because the target was standardised to zero mean and unit variance, Sigmoid and Tanh operate in their responsive, non-saturated range, so the vanishing-gradient problem that normally handicaps them does not appear on this simple one-feature task. The gaps are small and, with only six test points, partly within run-to-run variance. The honest conclusion is that **all three activations converge well and perform comparably**, with Tanh/Sigmoid marginally best here.

**Which loss function performed best?**
Holding the activation at ReLU, **MSE gave the strongest test accuracy (R² ≈ 0.870)**, ahead of Huber (≈ 0.861) and MAE (≈ 0.858). This is exactly what one would expect on clean data with no extreme outliers: MSE's quadratic penalty rewards squeezing down every residual, whereas MAE and Huber trade a little accuracy for a robustness the data does not actually need.

**Which combination produced the best regression performance?**
**Tanh + MSE** is selected as the **final model** — it achieved the best-tied R², and Tanh is zero-centred and stable. If the model were to be deployed on messier, real-world salary data containing outliers, **ReLU/Tanh + Huber** would be the safer production choice, sacrificing a fraction of R² in exchange for robustness.

**Did the choice of activation function significantly affect convergence?**
Modestly, not dramatically. All three activations reduced validation loss and flattened; ReLU reached the lowest scaled validation loss fastest, while Sigmoid and Tanh were slightly slower but landed at similar accuracy. On a deeper network or with larger, unscaled inputs the difference would be far more pronounced, as Sigmoid's saturation would then clearly slow learning.

**Did any loss function make the model more robust to outliers?**
Yes — **MAE and Huber**, by construction, penalise large residuals linearly rather than quadratically, so occasional extreme salaries influence training less than under MSE. That robustness simply was not rewarded on this particular dataset because it contains no strong outliers; on noisier salary data it would be an advantage.

**Overfitting or underfitting?**
With only about 24 training rows, the train-versus-validation gap is the key diagnostic. The ReLU + MSE model ends at roughly training loss ≈ 0.012 versus validation ≈ 0.055 (scaled): validation is higher but **flat rather than rising**, indicating mild — not severe — overfitting, which is typical for such a tiny dataset. No model shows the persistently high training *and* validation loss that would signal underfitting. Because the test set is so small, conclusions rest on the **shape** of the learning curves rather than on tiny differences in the metric table.

---

## 10. Final Model Selection and Conclusion

The final selected model is **Tanh + MSE**, which delivered the best-tied test R² (≈ 0.887) with a zero-centred, stable activation. The experiments show that on a clean, low-dimensional, monotonic dataset like this salary data:

- Activation choice matters only marginally once the target is properly scaled, because the task is simple enough that ReLU, Sigmoid and Tanh all converge to similar accuracy.
- MSE is the most accurate loss on outlier-free data, while MAE and Huber are the tools to reach for when robustness to outliers is required.
- The most defensible engineering choice depends on context: **Tanh/Sigmoid + MSE** for maximum accuracy on this clean data, or **Huber** if the model must generalise to noisier salary data with occasional extreme values.

The central lesson of the lab is reinforced: activation and loss functions should be chosen and justified according to the **characteristics of the data and the objective** — here, a standardised target that keeps bounded activations usable, and a clean, outlier-free signal that favours MSE — rather than selected arbitrarily and reported on accuracy alone.

---

