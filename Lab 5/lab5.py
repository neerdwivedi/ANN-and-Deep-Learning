# Practical 5 - CNN for Cats vs Dogs Classification

## Aim

To implement a Convolutional Neural Network (CNN) using TensorFlow/Keras for binary image classification of cats and dogs.

## Objective

The main objective of this practical is to understand how a CNN works with images and use it to classify an image into one of two classes - Cat or Dog.

## Libraries Used

- TensorFlow
- Keras
- Matplotlib
- OS
- Shutil

## Dataset

I used the Cats vs Dogs dataset containing two classes:

- Cat
- Dog

The images were stored in two separate folders:

    PetImages/
        Cat/
        Dog/

Before training, the images were checked because a few files in the dataset were invalid and could not be decoded properly.

Instead of deleting these files, I moved them to a separate `invalid_images` folder.

### Dataset Details

Initially, the dataset contained approximately 25,000 images.

During preprocessing:

- Invalid images found and moved: **7**
- Valid images remaining: **24,991**
- Training images: **19,993**
- Validation images: **4,998**

The data was divided using an **80:20 split**:

- 80% for training
- 20% for validation

## Image Preprocessing

All images in the dataset can have different sizes, so they were resized to:

    160 x 160 pixels

The images contain three colour channels:

    RGB = Red, Green, Blue

The original pixel values are between 0 and 255.

I used:

    Rescaling(1.0 / 255)

to convert the pixel values to the range 0 to 1. This makes the input easier for the neural network to process.

## Data Augmentation

Data augmentation was used to create slightly different versions of training images.

I used:

- Random horizontal flip
- Random rotation
- Random zoom

This helps the model learn from different variations of the same images and reduces overfitting.

## CNN Architecture

The CNN used in this practical was:

    Input Image (160 x 160 x 3)
                |
        Data Augmentation
                |
            Rescaling
                |
        Conv2D (32 filters)
                |
          MaxPooling2D
                |
        Conv2D (64 filters)
                |
          MaxPooling2D
                |
        Conv2D (128 filters)
                |
          MaxPooling2D
                |
             Flatten
                |
          Dropout (0.5)
                |
        Dense (128, ReLU)
                |
        Dense (1, Sigmoid)
                |
           Cat or Dog

### Conv2D

The convolution layers are used to detect features from images.

The earlier layers may detect simple features such as edges and shapes, while deeper layers can learn more complex patterns.

### MaxPooling2D

Max pooling reduces the size of the feature maps while keeping important information.

This also reduces the amount of computation required.

### Flatten

The output from the convolution layers is in multiple feature maps.

Flatten converts these feature maps into a single one-dimensional vector so that it can be given to the Dense layer.

### Dropout

I used a dropout value of 0.5.

During training, Dropout randomly ignores some neurons. This helps reduce overfitting.

### Dense Layer

A Dense layer with 128 neurons and ReLU activation was used after the convolution layers.

### Output Layer

The final layer contains only one neuron:

    Dense(1, activation="sigmoid")

This is because Cats vs Dogs is a **binary classification problem**.

The sigmoid function gives a value between 0 and 1.

In this model:

    Value <= 0.5 -> Cat
    Value > 0.5  -> Dog

## Model Compilation

The model was compiled using:

    Optimizer: Adam
    Loss Function: Binary Crossentropy
    Metric: Accuracy

### Why Binary Crossentropy?

Binary crossentropy is suitable because there are only two possible classes - Cat and Dog.

### Why Adam?

Adam automatically adjusts the learning process and is commonly used for training neural networks.

## Training

The CNN was trained for:

    Epochs = 10
    Batch Size = 32

An epoch means that the model has gone through the complete training dataset once.

A batch means that instead of processing every image together, the model processes a smaller group of images at a time.

## Results

The model improved steadily during training.

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|---|---:|---:|---:|---:|
| 1 | 63.22% | 69.67% | 0.6330 | 0.5804 |
| 2 | 71.56% | 77.31% | 0.5579 | 0.4908 |
| 3 | 74.78% | 79.51% | 0.5127 | 0.4446 |
| 4 | 76.08% | 81.43% | 0.4888 | 0.4145 |
| 5 | 78.12% | 81.81% | 0.4617 | 0.4044 |
| 6 | 78.99% | 83.03% | 0.4495 | 0.3917 |
| 7 | 79.97% | 82.73% | 0.4288 | 0.3851 |
| 8 | 80.77% | 83.27% | 0.4138 | 0.3758 |
| 9 | 82.02% | 83.65% | 0.3944 | 0.3752 |
| 10 | 82.80% | 84.45% | 0.3876 | 0.3526 |

### Final Result

After 10 epochs, the model achieved:

    Final Training Accuracy: 82.80%
    Final Validation Accuracy: 84.45%

    Final Training Loss: 0.3876
    Final Validation Loss: 0.3526

Therefore, the CNN was able to classify unseen Cat and Dog images with approximately **84.45% validation accuracy**.

## Observation

At the beginning of training, the model had a training accuracy of **63.22%** and validation accuracy of **69.67%**.

As the number of epochs increased, the accuracy generally increased and the loss decreased.

By the 10th epoch:

    Training Accuracy = 82.80%
    Validation Accuracy = 84.45%

The validation loss also decreased from:

    0.5804 -> 0.3526

This shows that the CNN was learning useful features from the images.

There was a small decrease in validation accuracy from epoch 6 to epoch 7, from **83.03% to 82.73%**, but it increased again in the later epochs. Small changes like this during training are normal.

The final validation accuracy was slightly higher than the training accuracy. One reason can be that data augmentation and dropout are applied during training, making the training task slightly harder.

Overall, the model showed good improvement over the 10 epochs and was able to classify most unseen Cat and Dog images correctly.

## Prediction

After training, the model was also used to predict images from the validation dataset.

For each image, the program displays:

- Predicted class
- Actual class
- Prediction confidence

The predictions were saved in:

    predictions.png

This makes it easy to visually compare the model's prediction with the actual class.

## Graphs Generated

The program generates the following files:

1. `example_images.png` - sample Cat and Dog images from the dataset
2. `accuracy_graph.png` - training and validation accuracy
3. `loss_graph.png` - training and validation loss
4. `predictions.png` - actual and predicted classes

The accuracy graph helps us see whether the model is improving.

The loss graph shows whether the error is decreasing during training.

## Note About Invalid Images

While checking the dataset, **7 invalid image files** were found.

These files could not be properly decoded by TensorFlow, so they were moved to a separate folder instead of being deleted.

Some other JPEG files produced warnings about extra JPEG data, but TensorFlow was still able to read them, so the training continued normally.

This cleaning step helped prevent invalid images from stopping the model during training.

## Conclusion

In this practical, I implemented a CNN using TensorFlow and Keras to classify images as Cats or Dogs.

I learned how convolution and pooling layers extract features from images and how preprocessing and data augmentation are used before training.

I also understood the use of ReLU, Sigmoid, Dropout, Binary Crossentropy and the Adam optimizer.

The model achieved a final validation accuracy of **84.45%**, showing that the CNN was able to learn useful features and perform binary image classification successfully.

## Learning Outcomes

After completing this practical, I was able to:

- Understand the basic architecture of a Convolutional Neural Network.
- Implement a CNN using TensorFlow/Keras.
- Perform image preprocessing and resizing.
- Apply data augmentation using flipping, rotation and zoom.
- Train a CNN for binary image classification.
- Evaluate the model using accuracy and loss.
- Interpret prediction results using predicted class and confidence.
- Understand the difference between training and validation performance.