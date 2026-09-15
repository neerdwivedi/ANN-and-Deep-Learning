"""
Email-style multiclass classifier on the 20 Newsgroups files
============================================================
Same idea as Gmail sorting mail into tabs. We take real messages that are already
labelled by topic and train a network to read a message and predict its topic.

We use 4 topics (think of them as 4 Gmail tabs):
    comp.graphics      -> "Tech"
    misc.forsale       -> "For sale"
    rec.sport.hockey   -> "Sports"
    sci.space          -> "Science"

The ONE new idea vs the Iris flowers: a message is words, not numbers, so first we
turn each message into numbers by counting important words (TF-IDF). After that, the
pipeline is identical: split -> build -> train -> test.

Run it from inside the Newsgroups folder:  python email_sorter.py
"""
import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# For similar results each time
np.random.seed(42)
tf.random.set_seed(42)



# 1. LOAD DATASET

categories = [
    "comp.graphics",
    "misc.forsale",
    "rec.sport.hockey",
    "sci.space"
]

class_names = [
    "Tech",
    "For Sale",
    "Sports",
    "Science"
]


# Download and load selected categories from 20 Newsgroups
dataset = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes")
)

texts = dataset.data
labels = dataset.target

print("Total messages:", len(texts))


# -------------------------------------------------------
# 2. CONVERT TEXT INTO NUMBERS USING TF-IDF
# -------------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words="english"
)

X = vectorizer.fit_transform(texts).toarray()

print("Number of input features:", X.shape[1])


# -------------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TESTING
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -------------------------------------------------------
# 4. BUILD THE MLP MODEL
# -------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])


# -------------------------------------------------------
# 5. COMPILE THE MODEL
# -------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()


# -------------------------------------------------------
# 6. TRAIN THE MODEL
# -------------------------------------------------------

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.10
)


# -------------------------------------------------------
# 7. EVALUATE THE MODEL
# -------------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")


# Predictions
predictions = model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)


# Classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0
    )
)


# Confusion matrix
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# -------------------------------------------------------
# 8. TEST WITH A NEW MESSAGE
# -------------------------------------------------------

new_message = [
    "NASA announced a new mission to send a rover to the moon next year."
]

new_message_vector = vectorizer.transform(new_message).toarray()

prediction = model.predict(
    new_message_vector,
    verbose=0
)[0]

predicted_class = np.argmax(prediction)


print("\nNew Message:")
print(new_message[0])

print("\nClass Probabilities:")

for class_name, probability in zip(class_names, prediction):
    print(f"{class_name}: {probability * 100:.2f}%")

print("\nPredicted Class:", class_names[predicted_class])