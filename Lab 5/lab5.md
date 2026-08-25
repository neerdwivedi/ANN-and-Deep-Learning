# Lab #5 — CNN for Cats vs Dogs 


---

## Aim

Build a simple Convolutional Neural Network (CNN) with TensorFlow/Keras that looks at a photo and decides whether it shows a **cat** or a **dog**. This is called **binary image classification** because there are exactly two possible answers.

---

## What is a CNN? (in plain words)

A CNN is a type of neural network designed for images. Instead of looking at every pixel on its own, it scans the picture with small "filters" that spot patterns. It builds up understanding in layers:

- The **first layers** notice simple things like edges and colours.
- The **middle layers** combine those into textures and shapes (like fur or an ear).
- The **last layers** use all of that to make the final decision: cat or dog.

The main building blocks used in this lab:

- **Conv2D layer** — slides small filters over the image to find patterns.
- **MaxPooling layer** — shrinks the image so the model keeps only the important information (and runs faster).
- **Flatten layer** — turns the picture data into one long list of numbers.
- **Dense layer** — an ordinary layer of neurons that makes the decision.
- **Sigmoid output** — one final neuron that gives a number between 0 and 1. Close to 0 means "cat", close to 1 means "dog".

---

## The Dataset

We use the **Cats vs Dogs (filtered)** dataset, which downloads automatically in the code.

- **2,000 training images** (1,000 cats + 1,000 dogs) — the model learns from these.
- **1,000 validation images** (500 cats + 500 dogs) — used to test if the model really learned, on pictures it never trained on.

The two classes are equal in number, so we can simply use **accuracy** (the percentage of correct guesses) to measure performance.

---

## Step-by-Step: What the Code Does

1. **Download and load the images.** Keras reads the pictures from the "cats" and "dogs" folders and labels them automatically.
2. **Preprocessing.** Every image is resized to **160×160** pixels and its colours are rescaled from 0–255 down to 0–1, which helps the network learn more smoothly.
3. **Data augmentation.** Because we only have 2,000 pictures, the model can easily just *memorise* them. To prevent this, we make small random changes to each training image — **flipping, rotating, and zooming** — so the model sees more variety and learns the real idea of "cat" and "dog" instead of memorising exact photos.
4. **Build the CNN.** Three Conv2D + MaxPooling blocks (with 32, 64, then 128 filters), followed by Flatten, a Dropout layer (which randomly ignores some values to avoid overfitting), a Dense layer, and the final sigmoid neuron.
5. **Compile and train.** We use the **adam** optimizer, **binary cross-entropy** loss (the correct choice for a two-class problem), and track **accuracy**. The model trains for 15 epochs (an epoch = one full pass through all training images).
6. **Plot learning curves.** We draw accuracy and loss for both training and validation across the epochs.
7. **Evaluate and predict.** We measure final accuracy on the validation set and show a grid of test pictures with the model's guess and confidence (correct guesses in green, wrong ones in red).

---

## Preprocessing and Data Augmentation (why they matter)

- **Resizing** makes every image the same size so they can be fed to the network together.
- **Rescaling to 0–1** keeps the numbers small, which makes training stable.
- **Augmentation** is the most important trick here: with such a small dataset, random flips/rotations/zooms act like "free extra data" and stop the model from overfitting (doing great on training pictures but badly on new ones).

---

## Results (fill in your own numbers after running)

When you run the notebook/script on Google Colab with a GPU, you will see something like:

| Measure | Typical value |
|---|---|
| Training accuracy | rises toward ~90%+ |
| **Validation accuracy** | **around 80–85%** |

Also look at the two learning curves:

- If **training accuracy** is much higher than **validation accuracy**, the model is **overfitting** (memorising). Data augmentation and dropout are what keep this gap small.
- If both accuracies rise together and level off, the model is learning well.

> Run the code, then write your real final accuracy and paste your three saved images (`training_curves.png`, `predictions.png`) into your submission.

---

## Interpreting the Predictions

The `predictions.png` grid shows the model's guesses:

- **Green titles** = correct guesses, usually clear, well-centred animals.
- **Red titles** = mistakes, usually blurry photos, odd angles, or animals that are far away or partly hidden — exactly the hard cases a small model struggles with.

The confidence percentage tells you how sure the model is. High confidence on correct pictures is a good sign; high confidence on wrong pictures shows where the model still needs to improve.

---

## Conclusion

A simple CNN can learn to tell cats from dogs quite well, even from only 2,000 pictures, reaching roughly 80–85% validation accuracy. The key lessons from this lab:

- CNNs read images layer by layer, from simple edges up to whole objects.
- Preprocessing (resize + rescale) and **data augmentation** are essential for good results on a small dataset.
- Accuracy and the learning curves together tell you whether the model is genuinely learning or just overfitting.

**Next step for higher accuracy:** use *transfer learning* — start from a model already trained on millions of images (like MobileNetV2) — which usually pushes accuracy above 95%.

---

