"""
Lab #5 - CNN for Cats vs Dogs (Beginner Version)
Student: Neer Dwivedi (24215217)

A simple Convolutional Neural Network (CNN) built with TensorFlow/Keras that
learns to tell apart pictures of CATS and DOGS.

HOW TO RUN
----------
1. Install the libraries (one time):
       pip install tensorflow matplotlib
2. Run the file:
       python lab5_cnn_simple.py

The dataset (about 68 MB) downloads automatically the first time.
Tip: this trains faster on Google Colab with a free GPU
     (Runtime -> Change runtime type -> GPU).
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Fixing a "seed" just makes results repeatable each run.
tf.random.set_seed(42)

# Every picture will be resized to this size (width, height).
IMG_SIZE = (160, 160)
BATCH = 32       # how many images the model looks at per step
EPOCHS = 15      # how many times the model sees the whole training set


# ---------------------------------------------------------------------------
# STEP 1: Download the pictures and load them
# ---------------------------------------------------------------------------
print("Step 1: downloading and loading the images...")

url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = keras.utils.get_file("cats_and_dogs_filtered.zip", origin=url, extract=True)
base_folder = os.path.join(os.path.dirname(zip_path), "cats_and_dogs_filtered")
train_folder = os.path.join(base_folder, "train")        # 2000 pictures
val_folder = os.path.join(base_folder, "validation")     # 1000 pictures

# Keras reads the images straight from the folders.
# "cats" folder -> label 0, "dogs" folder -> label 1.
train_data = keras.utils.image_dataset_from_directory(
    train_folder, image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary")
val_data = keras.utils.image_dataset_from_directory(
    val_folder, image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary")

class_names = train_data.class_names          # ['cats', 'dogs']
print("Classes the model will learn:", class_names)


# ---------------------------------------------------------------------------
# STEP 2: Look at a few example images (optional but helpful)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 8))
for images, labels in train_data.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.savefig("example_images.png")
plt.close()
print("Step 2: saved example_images.png")


# ---------------------------------------------------------------------------
# STEP 3: Build the CNN model
# ---------------------------------------------------------------------------
# The model is a stack of layers:
#   - Data augmentation: makes small random changes (flip/rotate/zoom) so the
#     model sees more variety and does not just memorise the training pictures.
#   - Rescaling: turns pixel values from 0-255 into 0-1 (networks like small numbers).
#   - Conv2D + MaxPooling (x3): the "eyes" of the CNN. Conv2D finds patterns
#     (edges, shapes, fur), MaxPooling shrinks the image to keep the important bits.
#   - Flatten: turns the 2D picture data into a single row of numbers.
#   - Dropout: randomly ignores some numbers while training to avoid overfitting.
#   - Dense layers: make the final decision.
#   - Last layer has 1 neuron with "sigmoid" -> outputs a probability (0 = cat, 1 = dog).
print("Step 3: building the CNN...")

model = keras.Sequential([
    keras.Input(shape=(160, 160, 3)),          # 160x160 colour image (3 = RGB)

    layers.RandomFlip("horizontal"),           # data augmentation
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    layers.Rescaling(1.0 / 255),               # scale pixels to 0-1

    layers.Conv2D(32, 3, activation="relu"),   # find simple patterns
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),   # find medium patterns
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),  # find complex patterns
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.5),                        # helps prevent overfitting
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid"),     # final answer: cat or dog
])

# "compile" = tell the model how to learn.
#   optimizer "adam"          -> the learning method
#   loss "binary_crossentropy"-> the right loss for a 2-class problem
#   metric "accuracy"         -> % of pictures it gets right
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()   # prints the list of layers


# ---------------------------------------------------------------------------
# STEP 4: Train the model
# ---------------------------------------------------------------------------
print("Step 4: training (this is the slow part)...")
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)


# ---------------------------------------------------------------------------
# STEP 5: Draw the learning curves
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy over time"); plt.xlabel("epoch"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss over time"); plt.xlabel("epoch"); plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()
print("Step 5: saved training_curves.png")


# ---------------------------------------------------------------------------
# STEP 6: Check how good the model is
# ---------------------------------------------------------------------------
loss, accuracy = model.evaluate(val_data)
print(f"\nFinal validation accuracy: {accuracy * 100:.1f}%")


# ---------------------------------------------------------------------------
# STEP 7: See some predictions (cat or dog + how confident)
# ---------------------------------------------------------------------------
print("Step 7: making predictions on a few pictures...")
images, labels = next(iter(val_data))
predictions = model.predict(images)

plt.figure(figsize=(12, 10))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.axis("off")
    prob = predictions[i][0]                       # closer to 1 = dog, closer to 0 = cat
    guess = "dogs" if prob > 0.5 else "cats"
    confidence = prob if guess == "dogs" else 1 - prob
    real = class_names[int(labels[i])]
    color = "green" if guess == real else "red"    # green = correct, red = wrong
    plt.title(f"Guess: {guess} ({confidence:.0%})\nReal: {real}", color=color)
plt.tight_layout()
plt.savefig("predictions.png")
plt.close()
print("Step 7: saved predictions.png")

print("\nAll done! Check these image files:")
print("  example_images.png, training_curves.png, predictions.png")
