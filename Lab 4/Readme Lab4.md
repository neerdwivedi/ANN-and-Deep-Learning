# Practical 4 - Keras MLP for Multiclass Classification

## Aim

To implement a Multi-Layer Perceptron (MLP) using Keras for a multiclass classification problem.

## What I Did

In this practical, I used a neural network to classify text messages into four different categories:

- Tech
- For Sale
- Sports
- Science

The data is taken from four categories of the 20 Newsgroups dataset.

Since a neural network cannot directly work with words, I first converted the text messages into numerical features using TF-IDF. These numerical features were then given as input to a simple MLP created using Keras.

## Dataset

The following four categories were used:

| Original Category | Class Name |
|---|---|
| comp.graphics | Tech |
| misc.forsale | For Sale |
| rec.sport.hockey | Sports |
| sci.space | Science |

The dataset is loaded directly using `fetch_20newsgroups()` from scikit-learn, so separate text files are not required.

## Steps

### 1. Load the Dataset

I loaded the four required categories from the 20 Newsgroups dataset.

Headers, footers and quoted text were removed so that the model mainly learns from the actual content of the messages.

### 2. Convert Text into Numbers

I used `TfidfVectorizer` with a maximum of 2000 features.

TF-IDF converts text into numerical values by giving importance to words based on how useful they are in identifying a message.

### 3. Split the Data

The dataset was divided into:

- 80% training data
- 20% testing data

I used `stratify=labels` so that all four classes are properly represented in both the training and testing sets.

### 4. Build the MLP

The neural network used was:

    2000 Input Features
            |
    64 Neurons (ReLU)
            |
    4 Neurons (Softmax)

The hidden layer contains 64 neurons with ReLU activation.

The output layer contains 4 neurons because there are four possible classes. Softmax is used in the output layer to generate probabilities for the four classes.

### 5. Compile and Train the Model

I used:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy
- Epochs: 10
- Batch Size: 32

Sparse categorical crossentropy is used because the target classes are represented as integer labels.

The model was trained for 10 epochs.

## Result Obtained

A total of **3934 messages** were loaded.

After splitting the dataset:

- Training samples: **3147**
- Testing samples: **787**
- Number of input features: **2000**

The final test accuracy obtained was:

**Test Accuracy = 89.96%**

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Tech | 0.899 | 0.872 | 0.885 | 195 |
| For Sale | 0.926 | 0.903 | 0.914 | 195 |
| Sports | 0.953 | 0.915 | 0.934 | 200 |
| Science | 0.829 | 0.909 | 0.867 | 197 |

Overall accuracy on 787 test messages was approximately **90%**.

### Confusion Matrix

    [[170   7   2  16]
     [  8 176   3   8]
     [  2   2 183  13]
     [  9   5   4 179]]

The diagonal values represent correctly classified messages.

For example:

- 170 Tech messages were correctly classified as Tech.
- 176 For Sale messages were correctly classified as For Sale.
- 183 Sports messages were correctly classified as Sports.
- 179 Science messages were correctly classified as Science.

## Testing with a New Message

I also tested the trained model using the following message:

    NASA announced a new mission to send a rover to the moon next year.

The model gave the following probabilities:

    Tech:      0.23%
    For Sale:  0.29%
    Sports:    1.12%
    Science:  98.37%

Therefore:

**Predicted Class = Science**

## Observation

The training accuracy increased from about **72.10% in the first epoch to 97.60% in the tenth epoch**.

The validation accuracy reached about **91.75%**.

When tested on completely unseen test data, the model achieved **89.96% accuracy**. This means that the model was able to correctly classify about 9 out of every 10 test messages.

Among the four classes, Sports had the highest precision of **0.953**. Science had a lower precision of **0.829**, which means that some messages belonging to other categories were also predicted as Science.

This can also be seen in the confusion matrix. For example, 16 Tech messages and 13 Sports messages were predicted as Science.

For the manually entered NASA message, the model predicted Science with **98.37% probability**, which is a sensible result because the message is related to a space mission.

Overall, the simple combination of TF-IDF and an MLP worked well for classifying these four different text topics.

## Conclusion

I successfully implemented a Keras MLP for multiclass text classification.

Through this practical, I understood that text first needs to be converted into numerical features before it can be given to a neural network. TF-IDF was used for this purpose.

I also understood that for multiclass classification, the output layer contains one neuron for each class and Softmax activation is used to obtain the probability of each class.

The model achieved a test accuracy of **89.96%** and correctly classified the sample NASA message as Science.

## Libraries Required

    pip install tensorflow scikit-learn numpy