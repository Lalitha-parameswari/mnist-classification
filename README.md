# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

MNIST Handwritten Digit Classification Dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

It is a widely used and deeply understood dataset and, for the most part, is “solved.” Top-performing models are deep learning convolutional neural networks that achieve a classification accuracy of above 99%, with an error rate between 0.4 %and 0.2% on the hold out test dataset.

![image](https://user-images.githubusercontent.com/103946827/191700149-c8fa02cb-0153-424a-a8aa-e2897e39e431.png)



## Neural Network Model

![image](https://user-images.githubusercontent.com/103946827/191700301-2f1b52da-e916-4714-9e0a-08e02498b6d7.png)


## DESIGN STEPS

### STEP 1:

Import the necessary packages and load the mnist dataset.

### STEP 2:

Consider 60,000 examples in the training dataset and 10,000 in the test dataset and that images are indeed square with 28×28 pixels.

### STEP 3:

Plot the first images in the dataset

### STEP 4:

use a one hot encoding for the class element of each sample, transforming the integer into a 10 element binary vector with a 1 for the index of the class value, and 0 values for all other classes.

### STEP 5:

Normalize the pixel values to 0/1.

### STEP 6:

Create a baseline convolutional neural network model for the problem and compile it.

### STEP 7:

plot the graph-Training Loss, Validation Loss Vs Iteration Plot

### STEP 8:

Evaluate the model through classification report and confusion matrix

### STEP 9:

Predict the output for the given handwritten input.

## PROGRAM

```
#Developed by: C.Lalitha Parameswari
#Registration number: 212219220027

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = image.load_img('img1.jpg')
type(img)
img = image.load_img('abc.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/103946827/191702529-38204b5b-b01a-4be9-a0bb-ed3aeeedfc2d.png)


### Classification Report

![image](https://user-images.githubusercontent.com/103946827/191702954-b5df1a45-b4a0-4409-aec1-e0072fc36eea.png)


### Confusion Matrix

![image](https://user-images.githubusercontent.com/103946827/191703267-09a44cac-eefa-4090-a6d9-bc54a8a872fc.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/103946827/191703467-be96f648-4b4b-4aed-85b3-542e21216bf5.png)


## RESULT

Thus, a convolutional deep neural network for digit classification is developed and the response for scanned handwritten images is verified.
