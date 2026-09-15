# Lab 7 - Keras MLP for Multiclass Classification

## Aim

To implement a Multi-Layer Perceptron (MLP) using TensorFlow/Keras for multiclass classification and compare the effect of different activation functions and optimizers on model performance.

---

## Dataset

I used the **Handwritten Digits dataset** available in scikit-learn using `load_digits()`.

The dataset contains small 8 x 8 grayscale images of handwritten digits from 0 to 9.

### Dataset Details

- Total samples: 1797
- Number of classes: 10
- Classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Input features: 64
- Image size: 8 x 8 pixels
- Missing values: 0

Each 8 x 8 image is represented using 64 pixel values, which are given as input to the MLP.

---

## Class Distribution

The dataset is almost balanced.

| Digit | Number of Samples |
|---|---:|
| 0 | 178 |
| 1 | 182 |
| 2 | 177 |
| 3 | 183 |
| 4 | 181 |
| 5 | 182 |
| 6 | 181 |
| 7 | 179 |
| 8 | 174 |
| 9 | 180 |

Since all the classes contain nearly the same number of samples, no special class balancing technique was required.

---

## Data Preprocessing

### 1. Train-Test Split

The dataset was divided into:

- 80% training data
- 20% testing data

This resulted in:

- Training samples: 1437
- Testing samples: 360

I used `stratify=y` so that the proportion of each digit remains similar in both training and testing data.

### 2. Feature Scaling

I used `StandardScaler` to scale the 64 input features.

The scaler was fitted only on the training data and then the same scaler was applied to the test data.

Scaling is useful because neural networks generally train more effectively when the input features are on a similar scale.

---

## MLP Architecture

The same basic architecture was used for all experiments so that the comparisons remain fair.

    Input Layer - 64 features
              |
    Hidden Layer 1 - 64 neurons
              |
    Hidden Layer 2 - 32 neurons
              |
    Output Layer - 10 neurons
              |
            Softmax

The activation function of the hidden layers was changed according to the experiment.

### Output Layer

The output layer contains 10 neurons because there are 10 possible classes.

I used:

    Softmax

as the output activation function.

Softmax converts the outputs into probabilities for the 10 classes. The class with the highest probability is selected as the predicted digit.

---

## Training Settings

The common training settings were:

- Epochs: 30
- Batch size: 32
- Validation split: 20% of training data
- Loss function: Sparse Categorical Crossentropy
- Output activation: Softmax

I used `sparse_categorical_crossentropy` because the target values are directly represented as integer labels from 0 to 9.

---

# Experiment A - Activation Function Comparison

In the first experiment, I compared three activation functions:

1. ReLU
2. Sigmoid
3. Tanh

The optimizer was kept fixed as **Adam**.

Therefore, the models were:

    ReLU + Adam
    Sigmoid + Adam
    Tanh + Adam

Keeping Adam fixed makes it easier to observe the effect of changing only the activation function.

### ReLU

ReLU outputs zero for negative values and keeps positive values.

It is widely used in neural networks because it is simple and generally allows fast training.

### Sigmoid

Sigmoid converts values into a range between 0 and 1.

It can sometimes learn more slowly in hidden layers because its gradients can become very small.

### Tanh

Tanh produces values between -1 and 1.

Unlike Sigmoid, it is zero-centered. In this experiment, Tanh gave the best overall performance.

---

# Experiment B - Optimizer Comparison

In the second experiment, I compared:

1. Adam
2. SGD
3. RMSprop

For this experiment, the hidden-layer activation function was kept fixed as **ReLU**.

Therefore, the models were:

    ReLU + Adam
    ReLU + SGD
    ReLU + RMSprop

The ReLU + Adam model was already trained in Experiment A, so it was reused instead of training the same model again.

### Adam

Adam automatically adjusts the learning process for individual parameters and usually provides fast convergence.

### SGD

Stochastic Gradient Descent updates the model parameters using the gradient of the loss.

It is simple, but it can require more training or tuning compared with adaptive optimizers.

### RMSprop

RMSprop also adapts the learning rate during training and generally performs well for neural networks.

---

# Results

All five models were evaluated on the **same 360 unseen test samples**.

Macro averaging was used for Precision, Recall and F1-score because this is a multiclass classification problem and the classes are almost balanced.

## Comparison Table

| Model | Test Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| ReLU + Adam | 97.22% | 0.9733 | 0.9717 | 0.9713 |
| Sigmoid + Adam | 95.56% | 0.9590 | 0.9551 | 0.9552 |
| **Tanh + Adam** | **98.89%** | **0.9893** | **0.9887** | **0.9887** |
| ReLU + SGD | 95.56% | 0.9581 | 0.9550 | 0.9546 |
| ReLU + RMSprop | 97.22% | 0.9738 | 0.9717 | 0.9717 |

---

# Best Model

The best-performing model was:

**Tanh + Adam**

Its results were:

    Test Accuracy = 98.89%
    Precision     = 0.9893
    Recall        = 0.9887
    F1-Score      = 0.9887

This model gave the highest overall performance among the five models tested.

---

# Confusion Matrix

The confusion matrix obtained for the best model, Tanh + Adam, was:

    [[36  0  0  0  0  0  0  0  0  0]
     [ 0 35  0  0  0  0  1  0  0  0]
     [ 0  0 35  0  0  0  0  0  0  0]
     [ 0  0  0 37  0  0  0  0  0  0]
     [ 0  0  0  0 36  0  0  0  0  0]
     [ 0  0  0  0  0 37  0  0  0  0]
     [ 0  0  0  0  0  0 36  0  0  0]
     [ 0  0  0  0  0  0  0 36  0  0]
     [ 0  2  0  1  0  0  0  0 32  0]
     [ 0  0  0  0  0  0  0  0  0 36]]

Out of 360 test images, only **4 images were incorrectly classified**.

Therefore:

    Correct predictions = 356
    Wrong predictions   = 4

This gives:

    356 / 360 = 98.89% accuracy

---

## Confusion Matrix Observation

The model classified most digits perfectly.

Digits:

    0, 2, 3, 4, 5, 6, 7 and 9

were classified correctly for every test sample.

Digit **1** was classified correctly 35 times, while one image of digit 1 was predicted as digit 6.

Digit **8** had the highest number of mistakes.

Out of 35 test images of digit 8:

- 32 were correctly predicted as 8
- 2 were predicted as 1
- 1 was predicted as 3

Therefore, **digit 8 was the most frequently misclassified class**.

This may happen because handwritten digits can have similar shapes. At a low resolution of only 8 x 8 pixels, some handwritten 8s may contain strokes that look similar to 1 or 3.

---

# Learning Curves

The program generated graphs to compare the models during training.

### Activation Comparison

File:

    activation_comparison.png

This graph compares the validation accuracy of:

- ReLU + Adam
- Sigmoid + Adam
- Tanh + Adam

It helps us understand how changing the hidden-layer activation function affects learning.

### Optimizer Comparison

File:

    optimizer_comparison.png

This compares:

- ReLU + Adam
- ReLU + SGD
- ReLU + RMSprop

It helps us compare the convergence of different optimizers.

### Best Model Accuracy

File:

    best_model_accuracy.png

This compares the training and validation accuracy of the final selected model.

### Best Model Loss

File:

    best_model_loss.png

This compares training and validation loss and can be used to check whether the model is learning properly or showing overfitting.

---

# Observation

All five MLP models were able to learn the handwritten-digit classification problem well, with test accuracies above 95%.

Changing the activation function affected the final performance.

With Adam fixed as the optimizer:

    ReLU    = 97.22%
    Sigmoid = 95.56%
    Tanh    = 98.89%

Tanh gave the best result in this experiment.

Changing the optimizer also affected performance.

With ReLU fixed as the activation:

    Adam    = 97.22%
    SGD     = 95.56%
    RMSprop = 97.22%

Adam and RMSprop performed better than SGD in this run.

The best overall combination was **Tanh + Adam**, which achieved **98.89% test accuracy**.

The confusion matrix also showed that the model generalized well because it made only 4 incorrect predictions out of 360 unseen test images.

---

# Analysis and Interpretation

## 1. Which activation function performed best? Why?

**Tanh performed best** in this experiment.

With Adam as the common optimizer:

- ReLU achieved 97.22% accuracy.
- Sigmoid achieved 95.56% accuracy.
- Tanh achieved 98.89% accuracy.

Tanh is zero-centered and worked well with the standardized input features in this dataset.

---

## 2. Which optimizer performed best? Why?

For the ReLU models, **Adam and RMSprop both achieved 97.22% test accuracy**.

RMSprop had a slightly higher F1-score:

    Adam F1    = 0.9713
    RMSprop F1 = 0.9717

Both performed better than SGD, which achieved 95.56% accuracy.

Adam and RMSprop adapt their updates during training, which can help them converge efficiently.

---

## 3. Which activation-function and optimizer combination gave the best performance?

The best combination was:

**Tanh + Adam**

It achieved:

- Test Accuracy: 98.89%
- Precision: 0.9893
- Recall: 0.9887
- F1-score: 0.9887

---

## 4. Did activation function affect convergence?

Yes.

The activation comparison graph shows that the models do not learn in exactly the same way.

ReLU and Tanh performed better than Sigmoid in the final test results.

Sigmoid can suffer from very small gradients when its values become saturated, which can make learning slower.

---

## 5. Did optimizer affect convergence speed?

Yes.

Adam and RMSprop are adaptive optimizers and generally reached good performance efficiently.

SGD produced the lowest test performance among the three ReLU models in this experiment.

---

## 6. Which model achieved the best validation performance?

The validation curves generated during training can be compared using `activation_comparison.png`, `optimizer_comparison.png`, and the best-model graphs.

The final model selected from the complete evaluation was **Tanh + Adam**.

---

## 7. Which model achieved the best test performance?

**Tanh + Adam** achieved the best test performance with **98.89% accuracy**.

---

## 8. Is there a significant difference between training and testing performance?

The test performance of the selected model is very high at 98.89%.

The training and validation curves should be viewed together with the test result to check the gap between them. A very large gap would suggest poor generalization, while similar performance indicates that the model generalizes well.

---

## 9. Does the model show overfitting or underfitting?

The generated training and validation curves are used to check this.

Overfitting would occur if training performance keeps improving while validation performance becomes worse.

Underfitting would occur if both training and validation performance remain poor.

The strong validation/test performance obtained in this experiment indicates that the selected MLP learned the digit classification task well.

---

## 10. Which class was most frequently misclassified?

Digit **8** was the most frequently misclassified class.

Three images of digit 8 were incorrectly classified:

- Two as digit 1
- One as digit 3

---

## 11. Why did these misclassifications occur?

The Digits dataset contains very small 8 x 8 grayscale images.

Different people write digits differently, and at such a low resolution some digits can have similar shapes.

For example, parts of a handwritten 8 may resemble the strokes of a 1 or 3.

---

## 12. What could improve the classification performance?

Possible improvements include:

- Training for more epochs if required
- Trying different learning rates
- Adding Dropout if overfitting occurs
- Changing the number of hidden neurons
- Using a larger handwritten-digit dataset
- Using a CNN instead of an MLP

A CNN can be especially useful for image data because it can learn spatial patterns between nearby pixels.

---

## 13. Which model should be selected as the final model?

I would select **Tanh + Adam**.

It achieved the highest:

- Test accuracy
- Precision
- Recall
- F1-score

It also made only 4 mistakes out of 360 test samples.

Therefore, it provided the best overall classification performance among the models tested.

---

# Effect of Dataset Characteristics

The characteristics of the dataset also affect the MLP.

### Number of Classes

There are 10 classes, so this is a multiclass classification problem.

Therefore, 10 output neurons with Softmax activation are used.

### Class Balance

The classes are almost equally distributed, so no class balancing technique was required.

### Feature Distribution

The input consists of 64 pixel intensity features.

StandardScaler was used to put the features on a suitable scale before training.

### Non-Linearity

Handwritten digits cannot be separated using only simple linear boundaries.

Hidden layers with nonlinear activation functions such as ReLU, Sigmoid and Tanh help the MLP learn more complex patterns.

### Overlapping Classes

Some handwritten digits can look similar, especially at the low 8 x 8 resolution.

This is visible in the confusion matrix, particularly for digit 8.

### Number of Samples

The dataset contains 1797 samples.

This is enough to train a small MLP for this practical while also keeping the training time low.

---

# Files Generated

The program generated the following files:

1. `confusion_matrix.png`
2. `activation_comparison.png`
3. `optimizer_comparison.png`
4. `best_model_accuracy.png`
5. `best_model_loss.png`

These graphs are useful for comparing the models and understanding their performance.

---

# Final Result

The MLP was successfully implemented for handwritten-digit multiclass classification.

Five different model configurations were compared.

The best model was:

**Tanh Activation + Adam Optimizer**

Final performance:

    Test Accuracy = 98.89%
    Precision     = 0.9893
    Recall        = 0.9887
    F1-Score      = 0.9887

Only **4 out of 360 test images were incorrectly classified**.

---

# Conclusion

In this practical, I implemented a Multi-Layer Perceptron using TensorFlow/Keras for handwritten-digit multiclass classification.

I compared three activation functions - ReLU, Sigmoid and Tanh - and three optimizers - Adam, SGD and RMSprop.

From the experiment, **Tanh + Adam gave the best overall result with 98.89% test accuracy**.

I also learned how accuracy, precision, recall, F1-score, confusion matrices and learning curves can be used together to evaluate a multiclass classification model instead of looking only at accuracy.

This practical helped me understand that the choice of activation function and optimizer can affect both the learning process and the final performance of an MLP.