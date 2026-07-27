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

import os
import re
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# The folder holding the .txt files (defaults to wherever this script lives)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# The 4 topic files we'll use, and a friendly name for each
CATEGORIES = {
    "comp.graphics.txt":    "Tech",
    "misc.forsale.txt":     "For sale",
    "rec.sport.hockey.txt": "Sports",
    "sci.space.txt":        "Science",
}
CLASS_NAMES = list(CATEGORIES.values())        # ['Tech', 'For sale', 'Sports', 'Science']


# ---------------------------------------------------------------------------
# STEP 1 - THE LOADER
# Each .txt file holds MANY messages stacked together; every message starts with a
# line "Newsgroup: ...". We split on that to get individual messages, then drop the
# header block (everything up to the first blank line) so the model reads the actual
# message text, not giveaway header fields.
# ---------------------------------------------------------------------------
def load_messages(path):
    text = open(path, encoding="latin-1").read()
    chunks = re.split(r"(?m)^Newsgroup:\s", text)      # split into separate messages
    messages = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        body = re.sub(r"(?s)^.*?\n\n", "", chunk, count=1)   # remove the header block
        if body.strip():
            messages.append(body)
    return messages


texts, labels = [], []
for label_index, (filename, friendly) in enumerate(CATEGORIES.items()):
    msgs = load_messages(os.path.join(DATA_DIR, filename))
    texts.extend(msgs)
    labels.extend([label_index] * len(msgs))           # this topic's number, repeated
    print(f"Loaded {len(msgs):5d} messages for '{friendly}'  ({filename})")

labels = np.array(labels)
print(f"\nTotal messages: {len(texts)}")


# ---------------------------------------------------------------------------
# STEP 2 - TURN WORDS INTO NUMBERS (the new part)
# TfidfVectorizer looks at all the messages and picks the 2000 most useful words,
# then turns each message into 2000 numbers: how strongly each of those words appears.
# 'stop_words' throws away filler like "the", "and". That row of 2000 numbers is what
# the network actually sees -- exactly like the 4 numbers per Iris flower, just more.
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
X = vectorizer.fit_transform(texts).toarray()          # shape: (num_messages, 2000)
print("Each message is now", X.shape[1], "numbers.")


# ---------------------------------------------------------------------------
# STEP 3 - SPLIT into train (80%) and test (20%) -- same as before
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train: {len(X_train)} messages    Test: {len(X_test)} messages")


# ---------------------------------------------------------------------------
# STEP 4 - BUILD THE NETWORK
# 2000 inputs -> hidden layer (64 neurons, ReLU) -> 4 outputs (softmax, one per topic)
# ---------------------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2000,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(4,  activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# ---------------------------------------------------------------------------
# STEP 5 - TRAIN
# ---------------------------------------------------------------------------
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=1)


# ---------------------------------------------------------------------------
# STEP 6 - TEST AND JUDGE
# ---------------------------------------------------------------------------
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\nTest accuracy: {test_acc:.2%}")

y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

print("Confusion matrix (rows = actual, cols = predicted):")
print(f"{'':<12}" + "".join(f"{n:>10}" for n in CLASS_NAMES))
for i, row in enumerate(confusion_matrix(y_test, y_pred)):
    print(f"{CLASS_NAMES[i]:<12}" + "".join(f"{v:>10}" for v in row))


# ---------------------------------------------------------------------------
# STEP 7 - TRY IT ON YOUR OWN MESSAGE
# ---------------------------------------------------------------------------
my_message = ["NASA announced a new mission to send a rover to the moon next year."]
my_numbers = vectorizer.transform(my_message).toarray()    # same word-to-number step
probs = model.predict(my_numbers, verbose=0)[0]
print("\nYour test message:", my_message[0])
for name, p in zip(CLASS_NAMES, probs):
    print(f"  {name:<10} {p:.1%}")
print(f"  -> best guess: {CLASS_NAMES[probs.argmax()]}")