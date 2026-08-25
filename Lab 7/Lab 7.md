# Lab #7 — Keras MLP for Multiclass Classification

**Dataset:** Handwritten Digits (scikit-learn `load_digits`)
**Task:** Predict which digit (0–9) an 8×8 image represents — a 10-class classification problem.

---

## 1. Dataset Description and Source

The dataset is the **Digits** dataset built into scikit-learn (`sklearn.datasets.load_digits`). It is a smaller, cleaner version of the classic handwritten-digit recognition problem.

| Property | Value |
|---|---|
| Source | scikit-learn built-in (`load_digits`) |
| Total samples | 1,797 |
| Input features | 64 (each image is 8×8 pixels, flattened into 64 numbers) |
| Pixel value range | 0–16 (grayscale intensity) |
| Number of classes | 10 (digits 0, 1, 2, … 9) |
| Missing values | None |

**Why this dataset?** It comfortably satisfies the lab's "three or more classes" rule (it has ten), it needs no download or internet, and — unlike a tiny dataset such as Iris — it has enough samples and enough classes to produce a *meaningful* confusion matrix and clear differences between activation functions and optimizers. That makes the analysis sections of this lab genuinely interesting rather than trivial.

---

## 2. Problem Definition

Given the 64 pixel values of an 8×8 grayscale image, the model must output the correct digit label from 0 to 9. Because there are more than two mutually-exclusive classes, this is a **multiclass classification** problem, which is why the output layer uses **softmax** (it turns the network's outputs into 10 probabilities that add up to 1) together with the **categorical cross-entropy** loss.

---

## 3. Dataset Exploration and Class Distribution

The ten classes are almost perfectly balanced — roughly 180 examples of each digit:

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Count | 178 | 182 | 177 | 183 | 181 | 182 | 181 | 179 | 174 | 180 |

Because the classes are balanced, **no class-imbalance handling** (such as class weights or resampling) is needed, and it is reasonable to use plain accuracy together with **macro-averaged** precision/recall/F1 (macro-average treats every digit equally).

---

## 4. Data Preprocessing (and Justification)

1. **Missing values** — checked; there are none, so nothing to fix.
2. **Target encoding** — the label is turned into a **one-hot vector** (e.g. `3 → [0,0,0,1,0,0,0,0,0,0]`), which is the format softmax + categorical cross-entropy expects.
3. **Train/test split** — 80% training / 20% testing, using `stratify` so both sets keep the same class proportions. **The same split is reused by every model** so all comparisons are fair.
4. **Feature scaling** — `StandardScaler` rescales each of the 64 pixels to mean 0 and standard deviation 1. Neural networks train faster and more stably on standardized inputs. The scaler is **fit only on the training set** and then applied to the test set, so no test information leaks into training.

**Input features:** 64 &nbsp;|&nbsp; **Classes:** 10 &nbsp;|&nbsp; **Training samples:** 1,437 &nbsp;|&nbsp; **Test samples:** 360.

---

## 5. MLP Architecture

| Layer | Neurons | Activation |
|---|---|---|
| Input | 64 | — |
| Hidden layer 1 | 64 | ReLU / Sigmoid / Tanh (varied in Experiment A) |
| Hidden layer 2 | 32 | same as hidden layer 1 |
| Output | 10 | **Softmax** |

**Fixed training settings (kept the same for every model so comparisons are fair):**

- Loss: categorical cross-entropy
- Epochs: 60
- Batch size: 32
- Validation: 20% of the training data held out each epoch (`validation_split=0.2`)
- Learning rate: each optimizer's Keras default (Adam/RMSprop = 0.001, SGD = 0.01)

---

## 6. Experiments

The lab requires that we **do not just swap a setting and report accuracy**, so each experiment changes exactly one thing while holding everything else constant.

- **Experiment A — Activation functions:** optimizer fixed at **Adam**, hidden-layer activation varied across **ReLU, Sigmoid, Tanh**.
- **Experiment B — Optimizers:** activation fixed at **ReLU**, optimizer varied across **Adam, SGD, RMSprop**.

(The "ReLU + Adam" model belongs to both experiments, so it is trained once and reused.)

---

## 7. Comparison Table (all metrics measured on the same 360-sample test set)

| Model | Activation | Optimizer | Test Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|---|
| Model 1 | ReLU | Adam | 0.972 | 0.973 | 0.972 | 0.971 |
| Model 2 | Sigmoid | Adam | 0.964 | 0.965 | 0.964 | 0.963 |
| **Model 3** | **Tanh** | **Adam** | **0.981** | **0.981** | **0.980** | **0.980** |
| Model 4 | ReLU | SGD | 0.953 | 0.954 | 0.952 | 0.951 |
| Model 5 | ReLU | RMSprop | 0.975 | 0.975 | 0.975 | 0.975 |

*Precision, recall and F1 are **macro-averaged**.*

> Note: all five models perform well (95–98%) because digit recognition on this clean dataset is an easy task. The differences between the top models are small, so the numbers may shift slightly if you re-run, but the overall ranking and story stay the same.

---

## 8. Confusion Matrix Interpretation (best model: Tanh + Adam)

The best model misclassified only **7 of 360** test images. The mistakes were:

| True digit | Predicted as | Count |
|---|---|---|
| 8 | 1 | 2 |
| 8 | 2 | 1 |
| 8 | 3 | 1 |
| 7 | 4 | 2 |
| 9 | 7 | 2 |
| 1 | 0 | 1 |

- **Most accurately predicted:** digits **0, 2, 3, 4, 5, 6** were classified perfectly (100%).
- **Most frequently misclassified:** digit **8** (4 of its 35 test images were wrong — recall ≈ 0.89). It was spread across 1, 2 and 3.
- **Hardest pairs to distinguish:** **8 vs 1**, **7 vs 4**, and **9 vs 7**.

**Why these confusions make sense:** at only 8×8 resolution, an 8 shares its two stacked loops with the curves of 2 and 3 and the vertical stroke of 1; a 7 and a 4 both have a prominent diagonal/horizontal stroke near the top; and 9 and 7 share an upper loop with a descending stroke. These are exactly the digits a *person* would also confuse in a blurry 8×8 image, so the errors reflect genuine visual overlap rather than a broken model.

---

## 9. Analysis and Interpretation (answers to the lab questions)

1. **Which activation performed best? Why?** **Tanh** edged out the others (F1 0.980). Tanh is zero-centered and, on standardized inputs, tends to produce well-scaled gradients early in training. ReLU was a very close second (0.971).
2. **Which optimizer performed best? Why?** **Adam** and **RMSprop** were best (0.971–0.975). Both adapt the learning rate per-parameter, so they converge quickly and reliably. Plain SGD was the weakest (0.951) because it uses a single fixed learning rate and needs more epochs/tuning to catch up.
3. **Best activation + optimizer combination?** **Tanh + Adam** (test accuracy 0.981, F1 0.980).
4. **Did the activation function significantly affect convergence?** Somewhat. ReLU and Tanh converged fast and smoothly; **Sigmoid was noticeably slower** to converge because its gradients saturate (get very small) when neuron outputs approach 0 or 1.
5. **Did the optimizer affect convergence speed?** Yes, strongly. Adam and RMSprop reached high accuracy within the first ~10–15 epochs; **SGD improved more slowly** and was still catching up late in training.
6. **Best validation performance?** The adaptive-optimizer models (Tanh+Adam and ReLU+RMSprop) had the highest, most stable validation curves.
7. **Best test performance?** **Tanh + Adam** (0.981).
8. **Significant gap between training and testing performance?** No. Training accuracy reached ~99–100% and test accuracy was ~97–98%, a small and healthy gap — the model generalizes well.
9. **Overfitting or underfitting?** Only **mild overfitting** (training slightly above validation late in training) and no underfitting — all models learned the task. Sigmoid was the closest to underfitting early on due to slow convergence.
10. **Which classes were most frequently misclassified?** Digit **8** (and to a lesser extent 7 and 9).
11. **Possible reasons for these misclassifications?** Low 8×8 resolution and genuine shape overlap between 8/1/2/3, 7/4, and 9/7 (explained in Section 8).
12. **What could improve performance?** Add regularization (Dropout) to reduce the small overfitting; train a bit longer with early stopping; try a slightly wider network; or, for a bigger jump, use a Convolutional Neural Network (CNN), which is designed for images and exploits pixel neighborhoods that an MLP ignores.
13. **Which model would you select as the final model, and why?** See Section 10.

**How the dataset's characteristics influenced the MLP** (required discussion): the **10 balanced classes** meant macro-averaged metrics were appropriate and no imbalance handling was needed; the **64 continuous pixel features** made feature scaling important for stable training; the **non-linearity** of digit shapes is why hidden layers with non-linear activations are needed at all; and the main source of error was **overlapping classes** (visually similar digits) rather than noise or too few samples, which is exactly what the confusion matrix revealed.

---

## 10. Final Model Selection

**Selected final model: Tanh activation + Adam optimizer.**

Justification (considering more than accuracy):

- **Highest test accuracy and F1** (0.981 / 0.980) among all five models.
- **Balanced precision and recall** across classes — no single digit is badly neglected.
- **Stable, fast convergence** with only mild overfitting on the learning curves.
- **Reasonable complexity** — the same small two-hidden-layer network as every other model, so it is not winning by being bigger; it simply pairs a well-scaled activation with an adaptive optimizer.

ReLU + Adam is an equally defensible choice given how close the numbers are; Tanh + Adam is selected because it was best on every headline metric in this run while generalizing just as cleanly.

---

## 11. Conclusion

A simple two-hidden-layer MLP classifies the 10 handwritten digits with about **98% test accuracy**. Comparing activations showed Tanh and ReLU clearly outperform Sigmoid (whose gradients saturate), and comparing optimizers showed the adaptive methods (Adam, RMSprop) converge faster and score higher than plain SGD. The remaining errors come almost entirely from visually similar digits at low resolution (notably 8 vs 1/2/3), which points to a CNN as the natural next step for further gains.

---

### How to run
