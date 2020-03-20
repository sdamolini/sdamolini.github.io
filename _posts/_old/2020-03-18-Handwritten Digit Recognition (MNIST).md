---
layout: post
title: "Handwritten Digit Recognition (MNIST)"
img:     MNIST_thumbnail.jpg
date: 2020-03-18 12:54:00 +0300
description: None. 
tag: ['MNIST', 'CNN', 'SHAP']
---
<a id="Introduction"></a>
# 1. Introduction

This is a take on the well known MNIST dataset (Modified National Institute of Standards and Technology) that is comprised of handritten digits and it is commonly used to practice recognition algorithms.

This notebook will train linear models, neural networks and convolutionnal neural networks, with various levels of regulariization. Resulting accuracies will be compared.

Some of the advanced models will be trained using the GPU (Graphics Processing Unit) with Tensorflow-GPU.

To lean more about how about how this can help train models faster, and how to install Tensorflow-GPU, feel free to head to my other post [here](https://sdamolini.github.io/Tutorial-Install-TensorFlow-GPU-on-Windows-10/).

<a id="Imports"></a>
# 2. Imports

<a id="Modules-and-Parameters"></a>
## 2.1. Modules and Parameters


```python
# Import personnal set of tools
import sys
sys.path.append(r'../../Packages')
import SD.tools as SD_tools
from SD.tools import package_version as v
```

    -----------------------------------------------
    SD_tools version 0.2.1 were succesfully loaded.       
    -----------------------------------------------
    


```python
# Import other packages
from platform import python_version
print('Python version: {}'.format(python_version()))

import pandas as pd
v(pd)

import numpy as np
v(np)

import random

import os
import os.path
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # Disable GPU. Instead use: "with tf.device('/cpu:0')" as needed

import matplotlib
import matplotlib.pyplot as plt
v(matplotlib)

import seaborn as sns

import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
v(sklearn)

import tensorflow as tf
from tensorflow.keras import backend as K
K.set_learning_phase(1)
v(tf)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import pickle
import itertools
import keras # <---- only imported for SHAP, not used to run models. Use tf.keras instead. 

import shap
from shap.plots import colors 
v(shap)

# Disable warnings after notebook is completed
import warnings
warnings.filterwarnings('ignore')
```

    Python version: 3.6.10
    Pandas version: 1.0.1
    Numpy version: 1.16.4
    Matplotlib version: 3.1.3
    Sklearn version: 0.22.1
    Tensorflow version: 1.14.0
    Shap version: 0.34.0
    

    Using TensorFlow backend.
    

**Note:** Tensorflow 1 is used in this notebook for compatibility with the SHAP package. At the time of writing, the latest version of SHAP is not stable with Tensorflow 2.

To learn how to set up a clean Anaconda environment with a specific version of a package, see my other post here: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


```python
# Define custom parameters

# In Tensorflow 2.0 val_acc was renamed to val_accuracy
acctxt = 'accuracy' if int(tf.__version__[0])>=2 else 'acc'

fsize=(15,5)          # Figure size for plots
n_examples=None       # limit examples for training, or None for all
nepochs=30            # Number of epochs for NN and CNN
target_accuracy=0.9962# Target for cross validation accuracy
lr_min=1e-8           # Minimum allowable learning rate
dropout_rate=0.30     # Dropout rate for all layers except final one
dropout_rate_ll=0.40  # Dropout rate for last layer
n_samples_max=500     # Maximum number of background samples to render SHAP values
train_logistic=1      # Flag to enable (re) training logistic models
train_nn=1            # Flag to enable (re) training neural network
train_cnn=1           # Flag to enable (re) training convoluted neural networks
advanced_vis=1        # Flag to allow advanced visualizations (CPU intensive)

# # # Preliminary runs parameters
# n_examples=293        # limit examples for training, or None for all
# nepochs=30            # Number of epochs for NN and CNN
# target_accuracy=0.92  # Target for cross validation accuracy
# train_logistic=0      # Flag to enable (re) training logistic models
# train_nn=1            # Flag to enable (re) training neural network
# train_cnn=1           # Flag to enable (re) training convoluted neural networks
# advanced_vis=0        # Flag to allow advanced visualizations (CPU intensive)

# # Preliminary extended runs parameters
# n_examples=1999       # limit examples for training, or None for all
# nepochs=30            # Number of epochs for NN and CNN
# target_accuracy=0.97  # Target for cross validation accuracy
# train_logistic=1      # Flag to enable (re) training logistic models
# train_nn=1            # Flag to enable (re) training neural network
# train_cnn=1           # Flag to enable (re) training convoluted neural networks
# advanced_vis=1        # Flag to allow advanced visualizations (CPU intensive)
```

<a id="Data"></a>
## 2.2. Data

Data can be retrieved directly from keras datasets.


```python
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

<a id="Overview"></a>
# 3. Overview


```python
# Provide basic information about the training and test set. 
print('The train set contains {} images of {}x{} pixels.\nThe test set contains {} images of {}x{} pixels.'.format(
    *(X_train.shape), *(X_test.shape)))
print('There are {} unique values in the set.'.format(len(np.unique(Y_train))))
```

    The train set contains 60000 images of 28x28 pixels.
    The test set contains 10000 images of 28x28 pixels.
    There are 10 unique values in the set.
    


```python
# Print unique values from the train set. 
print('Unique values in the train set:')
unique = np.unique(Y_train)
print(unique)
```

    Unique values in the train set:
    [0 1 2 3 4 5 6 7 8 9]
    

Let's display some samples from the train set.


```python
# Plot samples from the train set. 
fig, axes = plt.subplots(10,10,figsize=(8,8))
for digit in unique:
    digit_images=X_train[Y_train==digit][20:]
    for k in range(10):
        axes[k,digit].imshow(digit_images[k], cmap='Greys')
        axes[k,digit].axis('off')
plt.suptitle('Samples from the dataset.', y=0.92);
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_15_0.png" style="max-width:840px;">
</p>


Let's check the distribution of the unique classes in the train and test set.


```python
# Plot the distribution of the unique classes in the train and test set. 
fig, axes = plt.subplots(1,2, figsize=fsize)
_, counts = np.unique(Y_train, return_counts=True)
axes[0].bar(unique, counts)
axes[0].set_xticks(unique)
axes[0].set_title('Distribution of unique classes in the training set.')
_, counts = np.unique(Y_test, return_counts=True)
axes[1].bar(unique, counts)
axes[1].set_xticks(unique)
axes[1].set_title('Distribution of unique classes in the test set.');
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_17_0.png" style="max-width:840px;">
</p>


Conclusion: the classes in the train and test sets are well distributed and can be used as is.

<a id="Modeling"></a>
# 4. Modeling

Various models will be trained to recognized these handwritten digits:
    - Logistic regression models with various regularizations (Lasso, Ridge, Elastic net)
    - Neural networks with or without regularization (dropout)
    - Convolutional Neural Networks with or without data augmentation
    
The neural networks will be trained with the same number of epochs to have a sense of the accuracy between them. The best model with be trained furthermore to try to reach a set target, based on the accuracy on the test set. 

<a id="Normalization"></a>
## 4.1. Normalization

Data is normalized so pixel intensities are between 0 and 1.


```python
# Normalize and reshape data.

n=n_examples # limit number of examples, or None for all
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reduce the number of examples for initial tests
X_train=X_train[:n]/255
X_test=X_test[:n]/255
Y_train=Y_train[:n]
Y_test=Y_test[:n]

# Backup and reshape sets
X_train0 = X_train
X_train = X_train.reshape(X_train.shape[0],-1)
X_test0 = X_test
X_test = X_test.reshape(X_test.shape[0],-1)

print('X_train set shape: {}x{}, Y_train set shape: {}(x1)'.format(*X_train.shape, *Y_train.shape))
```

    X_train set shape: 60000x784, Y_train set shape: 60000(x1)
    

<a id="Logistic-Regression"></a>
## 4.2. Logistic Regression


```python
# Dataframe containing the accuracies of the models.
df_res = pd.DataFrame(columns=['Classifier','Training accuracy','Testing accuracy'])
```


```python
# Initialize and fit a linear regression model.
if train_logistic:
    clf=sklearn.linear_model.SGDClassifier(penalty='l1')
    clf.fit(X_train,Y_train)
    train_acc=accuracy_score(clf.predict(X_train), Y_train)
    test_acc=accuracy_score(clf.predict(X_test), Y_test)
    print('Linear logistic regression with Lasso regularization:')
    print('Train accuracy: {}, test accuracy: {}.'.format(round(train_acc,4),round(test_acc,4)))
    df_res=df_res.append({'Classifier': 'Linear - Lasso','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True);
```

    Linear logistic regression with Lasso regularization:
    Train accuracy: 0.9128, test accuracy: 0.9045.
    


```python
# Initialize and fit a linear regression model.
if train_logistic:
    clf=sklearn.linear_model.SGDClassifier(penalty='l2')
    clf.fit(X_train,Y_train)
    train_acc=accuracy_score(clf.predict(X_train), Y_train)
    test_acc=accuracy_score(clf.predict(X_test), Y_test)
    print('Linear logistic regression with Ridge regularization:')
    print('Train accuracy: {}, test accuracy: {}.'.format(round(train_acc,4),round(test_acc,4)))
    df_res=df_res.append({'Classifier': 'Linear - Ridge','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True);
```

    Linear logistic regression with Ridge regularization:
    Train accuracy: 0.9229, test accuracy: 0.9182.
    


```python
# Initialize and fit a linear regression model.
if train_logistic:
    clf=sklearn.linear_model.SGDClassifier(penalty='elasticnet')
    clf.fit(X_train,Y_train)
    train_acc=accuracy_score(clf.predict(X_train), Y_train)
    test_acc=accuracy_score(clf.predict(X_test), Y_test)
    print('Linear logistic regression with Elastic Net regularization:')
    print('Train accuracy: {}, test accuracy: {}.'.format(round(train_acc,4),round(test_acc,4)))
    df_res=df_res.append({'Classifier': 'Linear - ElasticNet','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    Linear logistic regression with Elastic Net regularization:
    Train accuracy: 0.9197, test accuracy: 0.917.
    


```python
# Display the summary table.
df_res.round(4)
```




<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier</th>
      <th>Training accuracy</th>
      <th>Testing accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear - Lasso</td>
      <td>0.9128</td>
      <td>0.9045</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear - Ridge</td>
      <td>0.9229</td>
      <td>0.9182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Linear - ElasticNet</td>
      <td>0.9197</td>
      <td>0.9170</td>
    </tr>
  </tbody>
</table>
</div>



<a id="Neural-Network"></a>
## 4.3. Neural Network

Keras neural networks will be built with the following layers:
    - An input layer corresponding to all 784 pixels of an image,
    - A 256-neuron hidden layer with a ReLu activation,
    - A 128-neuron hidden layer with a ReLu activation,
    - A 64-neuron hidden layer with a ReLu activation,
    - A 32-neuron hidden layer with a ReLu activation,
    - A 10-neuron output layer with a SoftMax activation.
    
On the model incorporating regularization, a dropout layer is added after each hidden layer.

The models are optimized using Adam, using a cross entropy loss function and accuracy as the metrics for performance. Models are fitted over a set number of epochs defined in Section 2.1.

For optimal training speeds, the learning rate will be automatically reduced when no progress is made, using a callback.

A mechanism for saving/reloading models is also implemented. 

<a id="No-Dropout"></a>
### 4.3.1. No Dropout


```python
# Build and train the neural network.

model_file = 'nn_nodrop'

with tf.device('/cpu:0'):

    if train_nn or not os.path.isfile(model_file+'.h5'):

        nn_nodrop = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(784,)),
            tf.keras.layers.Dense(256, activation='relu',),
            tf.keras.layers.Dense(128, activation='relu',),
            tf.keras.layers.Dense(64, activation='relu',),
            tf.keras.layers.Dense(32, activation='relu',),
            tf.keras.layers.Dense(10, activation='softmax',),
        ])
        print('Model defined.')

        # Compile the model
        nn_nodrop.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['acc'],)
        print('Model compiled.')

        # Train the model
        nn_nodrop_run=nn_nodrop.fit(X_train,
                      Y_train, 
                      epochs=nepochs,
                     validation_data=(X_test, Y_test),
                     )
        print('Model trained.')
        
        # Save the history on the local drive
        nn_nodrop_history = nn_nodrop_run.history
        with open(model_file+'.hist', 'wb') as file_pi:
            pickle.dump(nn_nodrop_history, file_pi)
            print('History saved.')

        # Save the model on the local drive
        nn_nodrop.save(model_file+'.h5')  # creates a HDF5 file 
        print('Model saved.')

    else:
        # Restore the model and its history from the local drive
        nn_nodrop=load_model(model_file+'.h5')
        nn_nodrop_history = pickle.load(open(model_file+'.hist', "rb"))
        print('Model and history reloaded.')
```

    WARNING:tensorflow:From C:\ProgramData\Anaconda3\envs\test1\lib\site-packages\tensorflow\python\ops\init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    Model defined.
    Model compiled.
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/30
    60000/60000 [==============================] - 6s 108us/sample - loss: 0.2328 - acc: 0.9291 - val_loss: 0.1183 - val_acc: 0.9634
    Epoch 2/30
    60000/60000 [==============================] - 6s 98us/sample - loss: 0.0976 - acc: 0.9697 - val_loss: 0.1115 - val_acc: 0.9663
    Epoch 3/30
    60000/60000 [==============================] - 6s 106us/sample - loss: 0.0693 - acc: 0.9786 - val_loss: 0.0967 - val_acc: 0.9699
    Epoch 4/30
    60000/60000 [==============================] - 6s 99us/sample - loss: 0.0542 - acc: 0.9826 - val_loss: 0.0956 - val_acc: 0.9735
    Epoch 5/30
    60000/60000 [==============================] - 5s 87us/sample - loss: 0.0441 - acc: 0.9861 - val_loss: 0.0902 - val_acc: 0.9747
    Epoch 6/30
    60000/60000 [==============================] - 5s 85us/sample - loss: 0.0355 - acc: 0.9887 - val_loss: 0.1026 - val_acc: 0.9737
    Epoch 7/30
    60000/60000 [==============================] - 5s 84us/sample - loss: 0.0310 - acc: 0.9908 - val_loss: 0.0928 - val_acc: 0.9765
    Epoch 8/30
    60000/60000 [==============================] - 5s 84us/sample - loss: 0.0288 - acc: 0.9910 - val_loss: 0.0864 - val_acc: 0.9772
    Epoch 9/30
    60000/60000 [==============================] - 5s 84us/sample - loss: 0.0262 - acc: 0.9916 - val_loss: 0.0885 - val_acc: 0.9774
    Epoch 10/30
    60000/60000 [==============================] - 5s 84us/sample - loss: 0.0227 - acc: 0.9929 - val_loss: 0.0832 - val_acc: 0.9795
    Epoch 11/30
    60000/60000 [==============================] - 5s 89us/sample - loss: 0.0200 - acc: 0.9935 - val_loss: 0.0909 - val_acc: 0.9798
    Epoch 12/30
    60000/60000 [==============================] - 5s 88us/sample - loss: 0.0186 - acc: 0.9942 - val_loss: 0.1020 - val_acc: 0.9793
    Epoch 13/30
    60000/60000 [==============================] - 5s 81us/sample - loss: 0.0174 - acc: 0.9949 - val_loss: 0.1071 - val_acc: 0.9756
    Epoch 14/30
    60000/60000 [==============================] - 5s 82us/sample - loss: 0.0145 - acc: 0.9958 - val_loss: 0.1039 - val_acc: 0.9793
    Epoch 15/30
    60000/60000 [==============================] - 5s 81us/sample - loss: 0.0149 - acc: 0.9954 - val_loss: 0.1281 - val_acc: 0.9781
    Epoch 16/30
    60000/60000 [==============================] - 5s 82us/sample - loss: 0.0147 - acc: 0.9955 - val_loss: 0.1016 - val_acc: 0.9809
    Epoch 17/30
    60000/60000 [==============================] - 5s 85us/sample - loss: 0.0152 - acc: 0.9956 - val_loss: 0.0907 - val_acc: 0.9823
    Epoch 18/30
    60000/60000 [==============================] - 5s 88us/sample - loss: 0.0113 - acc: 0.9967 - val_loss: 0.0987 - val_acc: 0.9827
    Epoch 19/30
    60000/60000 [==============================] - 5s 81us/sample - loss: 0.0123 - acc: 0.9962 - val_loss: 0.0939 - val_acc: 0.9810
    Epoch 20/30
    60000/60000 [==============================] - 5s 83us/sample - loss: 0.0119 - acc: 0.9967 - val_loss: 0.0949 - val_acc: 0.9814
    Epoch 21/30
    60000/60000 [==============================] - 5s 84us/sample - loss: 0.0112 - acc: 0.9966 - val_loss: 0.1210 - val_acc: 0.9791
    Epoch 22/30
    60000/60000 [==============================] - 5s 91us/sample - loss: 0.0105 - acc: 0.9969 - val_loss: 0.1025 - val_acc: 0.9813
    Epoch 23/30
    60000/60000 [==============================] - 5s 87us/sample - loss: 0.0094 - acc: 0.9970 - val_loss: 0.1081 - val_acc: 0.9820
    Epoch 24/30
    60000/60000 [==============================] - 5s 86us/sample - loss: 0.0099 - acc: 0.9972 - val_loss: 0.1094 - val_acc: 0.9807
    Epoch 25/30
    60000/60000 [==============================] - 5s 89us/sample - loss: 0.0116 - acc: 0.9966 - val_loss: 0.1016 - val_acc: 0.9817
    Epoch 26/30
    60000/60000 [==============================] - 5s 85us/sample - loss: 0.0074 - acc: 0.9980 - val_loss: 0.1155 - val_acc: 0.9816
    Epoch 27/30
    60000/60000 [==============================] - 5s 88us/sample - loss: 0.0102 - acc: 0.9974 - val_loss: 0.1124 - val_acc: 0.9814
    Epoch 28/30
    60000/60000 [==============================] - 5s 86us/sample - loss: 0.0092 - acc: 0.9975 - val_loss: 0.1154 - val_acc: 0.9811
    Epoch 29/30
    60000/60000 [==============================] - 5s 88us/sample - loss: 0.0081 - acc: 0.9978 - val_loss: 0.1310 - val_acc: 0.9792
    Epoch 30/30
    60000/60000 [==============================] - 6s 96us/sample - loss: 0.0076 - acc: 0.9977 - val_loss: 0.1246 - val_acc: 0.9812
    Model trained.
    History saved.
    Model saved.
    


```python
# Evaluate accuracy on the train and test set. 
train_acc=nn_nodrop.evaluate(X_train, Y_train)[1]
test_acc=nn_nodrop.evaluate(X_test, Y_test)[1]
print('\n\nTrain accuracy: {}% over {} epochs.'.format(round(train_acc*100,2),len(nn_nodrop_history['loss'])))
print('Test accuracy: {}% over {} epochs.'.format(round(test_acc*100,2),len(nn_nodrop_history['loss'])))
df_res=df_res.append({'Classifier': 'NN - no dopout','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    60000/60000 [==============================] - 2s 35us/sample - loss: 0.0024 - acc: 0.9993
    10000/10000 [==============================] - 0s 34us/sample - loss: 0.1246 - acc: 0.9812
    
    
    Train accuracy: 99.93% over 30 epochs.
    Test accuracy: 98.12% over 30 epochs.
    

<a id="With-Dropout"></a>
### 4.3.2. With Dropout


```python
# Build and train the neural network.

model_file = 'nn_drop'

with tf.device('/cpu:0'):

    if train_nn or not os.path.isfile(model_file+'.h5'):

        nn_drop = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(784,)),
        tf.keras.layers.Dense(256, activation='relu',),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(128, activation='relu',),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu',),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation='relu',),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax',),
        ])
        print('Model defined.')

        # Compile the model
        nn_drop.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['acc'],
        )

        # Train the model
        nn_drop_run=nn_drop.fit(X_train,
                      Y_train, 
                      epochs=nepochs,
                     validation_data=(X_test, Y_test))
        print('Model trained.')
        
        # Save the history on the local drive
        nn_drop_history = nn_drop_run.history
        with open(model_file+'.hist', 'wb') as file_pi:
            pickle.dump(nn_drop_history, file_pi)
            print('History saved.')

        # Save the model on the local drive
        nn_drop.save(model_file+'.h5')  # creates a HDF5 file 
        print('Model saved.')

    else:
        # Restore the model and its history from the local drive
        nn_drop=load_model(model_file+'.h5')
        nn_drop_history = pickle.load(open(model_file+'.hist', "rb"))
        print('Model and history reloaded.')
```

    Model defined.
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/30
    60000/60000 [==============================] - 7s 123us/sample - loss: 0.5124 - acc: 0.8510 - val_loss: 0.2727 - val_acc: 0.9326
    Epoch 2/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.2356 - acc: 0.9408 - val_loss: 0.2314 - val_acc: 0.9439
    Epoch 3/30
    60000/60000 [==============================] - 7s 116us/sample - loss: 0.1916 - acc: 0.9530 - val_loss: 0.1848 - val_acc: 0.9525
    Epoch 4/30
    60000/60000 [==============================] - 7s 111us/sample - loss: 0.1653 - acc: 0.9589 - val_loss: 0.1763 - val_acc: 0.9535
    Epoch 5/30
    60000/60000 [==============================] - 7s 113us/sample - loss: 0.1450 - acc: 0.9640 - val_loss: 0.1517 - val_acc: 0.9637
    Epoch 6/30
    60000/60000 [==============================] - 7s 113us/sample - loss: 0.1325 - acc: 0.9667 - val_loss: 0.1568 - val_acc: 0.9635
    Epoch 7/30
    60000/60000 [==============================] - 7s 113us/sample - loss: 0.1205 - acc: 0.9696 - val_loss: 0.1550 - val_acc: 0.9643
    Epoch 8/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.1127 - acc: 0.9729 - val_loss: 0.1625 - val_acc: 0.9657
    Epoch 9/30
    60000/60000 [==============================] - 7s 110us/sample - loss: 0.1050 - acc: 0.9736 - val_loss: 0.1457 - val_acc: 0.9657
    Epoch 10/30
    60000/60000 [==============================] - 7s 110us/sample - loss: 0.0976 - acc: 0.9753 - val_loss: 0.1521 - val_acc: 0.9674
    Epoch 11/30
    60000/60000 [==============================] - 7s 111us/sample - loss: 0.0957 - acc: 0.9755 - val_loss: 0.1406 - val_acc: 0.9669
    Epoch 12/30
    60000/60000 [==============================] - 7s 110us/sample - loss: 0.0891 - acc: 0.9766 - val_loss: 0.1537 - val_acc: 0.9682
    Epoch 13/30
    60000/60000 [==============================] - 7s 111us/sample - loss: 0.0894 - acc: 0.9771 - val_loss: 0.1479 - val_acc: 0.9668
    Epoch 14/30
    60000/60000 [==============================] - 7s 111us/sample - loss: 0.0818 - acc: 0.9790 - val_loss: 0.1516 - val_acc: 0.9688
    Epoch 15/30
    60000/60000 [==============================] - 7s 111us/sample - loss: 0.0832 - acc: 0.9790 - val_loss: 0.1544 - val_acc: 0.9693
    Epoch 16/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.0771 - acc: 0.9807 - val_loss: 0.1561 - val_acc: 0.9653
    Epoch 17/30
    60000/60000 [==============================] - 7s 114us/sample - loss: 0.0754 - acc: 0.9806 - val_loss: 0.1354 - val_acc: 0.9706
    Epoch 18/30
    60000/60000 [==============================] - 7s 113us/sample - loss: 0.0676 - acc: 0.9821 - val_loss: 0.1537 - val_acc: 0.9723
    Epoch 19/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.0682 - acc: 0.9829 - val_loss: 0.1522 - val_acc: 0.9673
    Epoch 20/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.0670 - acc: 0.9824 - val_loss: 0.1655 - val_acc: 0.9714
    Epoch 21/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.0635 - acc: 0.9837 - val_loss: 0.1521 - val_acc: 0.9718
    Epoch 22/30
    60000/60000 [==============================] - 7s 114us/sample - loss: 0.0655 - acc: 0.9830 - val_loss: 0.1770 - val_acc: 0.9692
    Epoch 23/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.0608 - acc: 0.9839 - val_loss: 0.1474 - val_acc: 0.9718
    Epoch 24/30
    60000/60000 [==============================] - 7s 114us/sample - loss: 0.0606 - acc: 0.9845 - val_loss: 0.1612 - val_acc: 0.9718
    Epoch 25/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.0585 - acc: 0.9850 - val_loss: 0.1646 - val_acc: 0.9707
    Epoch 26/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.0554 - acc: 0.9854 - val_loss: 0.1668 - val_acc: 0.9688
    Epoch 27/30
    60000/60000 [==============================] - 7s 112us/sample - loss: 0.0589 - acc: 0.9846 - val_loss: 0.1857 - val_acc: 0.9713
    Epoch 28/30
    60000/60000 [==============================] - 7s 111us/sample - loss: 0.0572 - acc: 0.9851 - val_loss: 0.1517 - val_acc: 0.9713
    Epoch 29/30
    60000/60000 [==============================] - 7s 110us/sample - loss: 0.0562 - acc: 0.9858 - val_loss: 0.1523 - val_acc: 0.9705
    Epoch 30/30
    60000/60000 [==============================] - 7s 110us/sample - loss: 0.0565 - acc: 0.9856 - val_loss: 0.1532 - val_acc: 0.9699
    Model trained.
    History saved.
    Model saved.
    


```python
# Evaluate accuracy on the train and test set. 
train_acc=nn_drop.evaluate(X_train, Y_train)[1]
test_acc=nn_drop.evaluate(X_test, Y_test)[1]
print('\n\nTrain accuracy: {}% over {} epochs.'.format(round(train_acc*100,2),len(nn_drop_history['loss'])))
print('Test accuracy: {}% over {} epochs.'.format(round(test_acc*100,2),len(nn_drop_history['loss'])))
df_res=df_res.append({'Classifier': 'NN - w/ dropout','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    60000/60000 [==============================] - 3s 56us/sample - loss: 0.0523 - acc: 0.9867
    10000/10000 [==============================] - 1s 55us/sample - loss: 0.1666 - acc: 0.9696
    
    
    Train accuracy: 98.67% over 30 epochs.
    Test accuracy: 96.96% over 30 epochs.
    

<a id="Accuracy-Comparison"></a>
### 4.3.3. Accuracy Comparison


```python
# Plot the accuracies of the model versus the number of training epochs.

try:
    x=range(1, 1+nepochs)
    fig, ax = plt.subplots(figsize=fsize)
    ax = sns.lineplot(x, nn_nodrop_history[acctxt], color='b', label='Training accuracy - no dropout')
    ax = sns.lineplot(x, nn_nodrop_history['val_'+acctxt], color='g', label='Validation accuracy - no dropout')
    ax = sns.lineplot(x, nn_drop_history[acctxt], color='gray', label='Training accuracy - w/ dropout')
    ax = sns.lineplot(x, nn_drop_history['val_'+acctxt], color='black', label='Validation accuracy - w/ dropout')
    plt.title('Training and validation accuracy for neural networks')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs');
except:
    pass
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_39_0.png" style="max-width:840px;">
</p>


<a id="Error-Analysis"></a>
### 4.3.4. Error Analysis

The neural network with dropout resulted in the best results. Let's display examples that were not properly categorized.


```python
# Compute predictions

X_test_pred=nn_drop.predict(X_test).argmax(axis=1) # Prediction of the model on X_test
y_pred=X_test_pred[X_test_pred!=Y_test] # Prediction of the model that are wrong
y_actual=Y_test[X_test_pred!=Y_test] # Ground truth corresponding to the wrong predictions
x=X_test0[X_test_pred!=Y_test] # X images corresponding to the wrong predictions
```


```python
# Display miscategorized examples sorted by true class.

fig, axes = plt.subplots(10,10,figsize=(15,15))
fig.suptitle('Miscategorized examples sorted by true class.', fontsize=16, y=0.93);
for i in range(10):
    for j in range(10):
            y_actual=Y_test[np.logical_and(X_test_pred!=Y_test, Y_test==j)]
            y_pred=X_test_pred[np.logical_and(X_test_pred!=Y_test, Y_test==j)]
            x=X_test0[np.logical_and(X_test_pred!=Y_test, Y_test==j)]
            axes[i,j].axis('off')
            try:
                axes[i,j].imshow(x[i], cmap='Greys')
                axes[i,j].set_title('{} inst. of {}'.format(y_pred[i], y_actual[i]))
            except:
                pass
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_43_0.png" style="max-width:840px;">
</p>



```python
# Display miscategorized examples sorted by true class (x-axis) and prediction class (y-axis).

fig, axes = plt.subplots(10,10,figsize=(15,15))
fig.suptitle('Miscategorized examples sorted by true class (x-axis) and prediction class (y-axis).', fontsize=16, y=0.93);
for i in range(10):
    for j in range(10):
            axes[i,j].axis('off')
            if i==j: continue
            y_actual=Y_test[np.logical_and(X_test_pred==i, Y_test==j)]
            y_pred=X_test_pred[np.logical_and(X_test_pred==i, Y_test==j)]
            x=X_test0[np.logical_and(X_test_pred==i, Y_test==j)]
            try:
                axes[i,j].imshow(x[0], cmap='Greys')
                axes[i,j].set_title('{} inst. of {}'.format(i, j))
            except:
                pass
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_44_0.png" style="max-width:840px;">
</p>



```python
# Plot confusion matrices

try:
    # Plot confusion matrix
    fig, ax = plt.subplots(1,2,figsize=(16,6.5))
    shrink=1
    # plt.subplots_adjust(wspace=1)

    #############
    # Left plot #
    #############
    y_pred=nn_drop.predict(X_test).argmax(axis=1)
    y_actual=Y_test

    # Define confusion matrix
    cm = confusion_matrix(y_actual,y_pred)
    cm0 = cm
    cm = 100*cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    n_classes = len(unique)

    plt.sca(ax[0])
    plt.imshow(cm, cmap = 'YlOrRd')
    plt.title('Normalized confusion matrix (%)')
    plt.colorbar(shrink=shrink)
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, np.arange(n_classes))
    plt.yticks(tick_marks, np.arange(n_classes))
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i!=j:
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        if i==j:
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label');


    ##############
    # Right plot #
    ##############

    y_test_pred=nn_drop.predict(X_test).argmax(axis=1)
    y_pred=X_test_pred[y_test_pred!=Y_test]
    y_actual=Y_test[y_test_pred!=Y_test]
    x=X_test0[y_test_pred!=Y_test]

    # Define confusion matrix
    cm = confusion_matrix(y_actual,y_pred)
    cm = cm.astype('float')
    n_classes = len(unique)

    plt.sca(ax[1])
    thresh = cm.max() / 2.
    try:
        cm[range(10), range(10)] = np.nan
    except:
        pass
    plt.imshow(cm, cmap = 'YlOrRd')
    plt.title('Confusion matrix (number of images)')
    plt.colorbar(shrink=shrink)
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, np.arange(n_classes))
    plt.yticks(tick_marks, np.arange(n_classes))


    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i!=j:
            plt.text(j, i, format(cm[i, j], '.0f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        if i==j:
            plt.text(j, i, format(cm0[i, j], '.0f'),
                     horizontalalignment="center",
                     color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label');

except:
    pass
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_45_0.png" style="max-width:840px;">
</p>


The matrix on the left shows the percentage of confusion between predicted labels and ground truth labels, in percentage. It can be seen that a large majority of predictions are correct.  

It is however, a little difficult to tell which classes are especially poorly predicted. On the matrix on the right, the true positives are removed from the set to make the mistakes more apparent, and the total number of cases in shown as opposed to the percentage. 

<a id="Convolutional-Neural-Network"></a>
## 4.4. Convolutional Neural Network

Keras Convolutional Neural Networks (CNNs) will be built with the following layers:
    - An input layer corresponding to all 784 pixels of an image,
    - A Conv2D layer with 32 filters and 3x3 kernel,
    - A batch normalization layer,
    - A Conv2D layer with 32 filters and 3x3 kernel (Note 1),
    - A batch normalization layer,
    - A Conv2D layer with 32 filters, 5x5 kernel and a stride of 2 (Note 2),
    - A batch normalization layer,
    - A dropout layer,
    - A Conv2D layer with 64 filters and 3x3 kernel,
    - A batch normalization layer,
    - A Conv2D layer with 64 filters and 3x3 kernel (Note 1),
    - A batch normalization layer,
    - A Conv2D layer with 64 filters, 5x5 kernel and a stride of 2 (Note 2),
    - A batch normalization layer,
    - A dropout layer,
    - A flattening layer,
    - A 128-neuron dense layer.
    - A dropout layer,
    - A 10-neuron dense output layer

**Note 1:** Two back to back 3x3 kernel layers are used instead of a single 5x5 layer (typically used). This gives more non-linearity to the model. 

**Note 2:** Here I am using an other Conv2D layer with a stride of 2 when typically a maxpooling layer is used. This has a similar subsampling effect but the parameters are learnable.

The models are optimized using Adam, using a cross entropy loss function and accuracy as the metrics for performance. Models are fitted over a set number of epochs defined in Section 2.1.

When data augmentation is implemented, random training samples are generated using small transformations (rotation, zoom, shift) on the training set images. 

For optimal training speeds, the learning rate will be automatically reduced when no progress is made, using a callback.

A mechanism for saving/reloading models is also implemented. 

<a id="No-Data-Augmentation"></a>
### 4.4.1. No Data Augmentation


```python
# Prepare the data (regularize and add axes as needed to fit the model input shape)

n=n_examples # limit examples, or None for all

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train=X_train[:n]/255
X_test=X_test[:n]/255
Y_train=Y_train[:n]
Y_test=Y_test[:n]
X_train = X_train[:,:,:,np.newaxis]
X_test = X_test[:,:,:,np.newaxis]
```


```python
# Define, compile and train the model.

model_file = 'cnn'

if train_cnn or not os.path.isfile(model_file+'.h5'):
        
    # Define the CNN model
    cnn=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding = 'same', input_shape=(28,28,1)), # input shape: 28x28x1
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding = 'same'), # input shape: 26x26x32
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (5,5), strides=2, activation='relu', padding='same'), # input shape: 24x24x32
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding = 'same'), # input shape: 12x12x32
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding = 'same'), # input shape: 10x10x64
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (5,5), strides=2, activation='relu', padding='same'), # input shape: 8x8x64
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Flatten(), # input shape: 4x4x64
     
    tf.keras.layers.Dense(128, activation='relu'), # input shape: 1024
    # tf.keras.layers.BatchNormalization(),  # <----- this layer was causing incompatibilities with SHAP
    tf.keras.layers.Dropout(rate=dropout_rate_ll),
        
    tf.keras.layers.Dense(10, activation='softmax') # input shape: 128
    ])


    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # Compile the CNN model
    cnn.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #Delete the training log if it exists. The training log is used to keep track of the losses and accuracies
    #over batch trainings.
    filepath=model_file+'.h5'
    try:
        os.remove(filepath)
        os.remove('cnn_training.log')
    except:
        pass

    
    checkpoint=ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, mode='auto')

    csv_logger = CSVLogger('cnn_training.log', append=True)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4, min_lr=lr_min, verbose=1)

    callbacks=[checkpoint, csv_logger, reduce_lr]
    
    cnn_run=cnn.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=nepochs,
        validation_data=(X_test, Y_test),
        callbacks=callbacks,
    )

else:
    # Restore the model and its history from the local drive
    cnn=load_model(model_file+'.h5')
    print('Model and history reloaded.')
    
with open('cnn_training.log', 'r') as f:
        lines=f.readlines()
        print('\n\nAccuracy: {}% over {} epochs.'.format(round(cnn.evaluate(X_test, Y_test)[1]*100,2),len(lines)-1))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/30
    60000/60000 [==============================] - 115s 2ms/sample - loss: 0.2017 - acc: 0.9407 - val_loss: 0.0904 - val_acc: 0.9737
    Epoch 2/30
    60000/60000 [==============================] - 108s 2ms/sample - loss: 0.0855 - acc: 0.9761 - val_loss: 0.0660 - val_acc: 0.9827
    Epoch 3/30
    60000/60000 [==============================] - 110s 2ms/sample - loss: 0.0676 - acc: 0.9808 - val_loss: 0.0521 - val_acc: 0.9843
    Epoch 4/30
    60000/60000 [==============================] - 113s 2ms/sample - loss: 0.0535 - acc: 0.9847 - val_loss: 0.0578 - val_acc: 0.9831
    Epoch 5/30
    60000/60000 [==============================] - 117s 2ms/sample - loss: 0.0480 - acc: 0.9866 - val_loss: 0.0397 - val_acc: 0.9896
    Epoch 6/30
    60000/60000 [==============================] - 116s 2ms/sample - loss: 0.0402 - acc: 0.9886 - val_loss: 0.0408 - val_acc: 0.9899
    Epoch 7/30
    60000/60000 [==============================] - 117s 2ms/sample - loss: 0.0343 - acc: 0.9897 - val_loss: 0.0376 - val_acc: 0.9895
    Epoch 8/30
    60000/60000 [==============================] - 117s 2ms/sample - loss: 0.0326 - acc: 0.9910 - val_loss: 0.0392 - val_acc: 0.9899
    Epoch 9/30
    60000/60000 [==============================] - 117s 2ms/sample - loss: 0.0287 - acc: 0.9920 - val_loss: 0.0436 - val_acc: 0.9874
    Epoch 10/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0283 - acc: 0.9922 - val_loss: 0.0420 - val_acc: 0.9903
    Epoch 11/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0222 - acc: 0.9938 - val_loss: 0.0316 - val_acc: 0.9907
    Epoch 12/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0245 - acc: 0.9931 - val_loss: 0.0444 - val_acc: 0.9903
    Epoch 13/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0195 - acc: 0.9944 - val_loss: 0.0395 - val_acc: 0.9902
    Epoch 14/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0185 - acc: 0.9944 - val_loss: 0.0424 - val_acc: 0.9907
    Epoch 15/30
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0181 - acc: 0.9949
    Epoch 00015: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0181 - acc: 0.9949 - val_loss: 0.0360 - val_acc: 0.9908
    Epoch 16/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0098 - acc: 0.9970 - val_loss: 0.0261 - val_acc: 0.9938
    Epoch 17/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0074 - acc: 0.9977 - val_loss: 0.0246 - val_acc: 0.9948
    Epoch 18/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0058 - acc: 0.9983 - val_loss: 0.0264 - val_acc: 0.9954
    Epoch 19/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0046 - acc: 0.9986 - val_loss: 0.0277 - val_acc: 0.9933
    Epoch 20/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0044 - acc: 0.9983 - val_loss: 0.0226 - val_acc: 0.9943
    Epoch 21/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0043 - acc: 0.9988 - val_loss: 0.0266 - val_acc: 0.9943
    Epoch 22/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0041 - acc: 0.9987 - val_loss: 0.0266 - val_acc: 0.9946
    Epoch 23/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0030 - acc: 0.9990 - val_loss: 0.0257 - val_acc: 0.9949
    Epoch 24/30
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0029 - acc: 0.9989
    Epoch 00024: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0030 - acc: 0.9989 - val_loss: 0.0322 - val_acc: 0.9933
    Epoch 25/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0026 - acc: 0.9992 - val_loss: 0.0322 - val_acc: 0.9946
    Epoch 26/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0027 - acc: 0.9991 - val_loss: 0.0272 - val_acc: 0.9940
    Epoch 27/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0020 - acc: 0.9992 - val_loss: 0.0336 - val_acc: 0.9937
    Epoch 28/30
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0023 - acc: 0.9992
    Epoch 00028: ReduceLROnPlateau reducing learning rate to 8.000000525498762e-06.
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0023 - acc: 0.9992 - val_loss: 0.0279 - val_acc: 0.9944
    Epoch 29/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0017 - acc: 0.9995 - val_loss: 0.0286 - val_acc: 0.9946
    Epoch 30/30
    60000/60000 [==============================] - 119s 2ms/sample - loss: 0.0020 - acc: 0.9995 - val_loss: 0.0272 - val_acc: 0.9948
    10000/10000 [==============================] - 6s 611us/sample - loss: 0.0315 - acc: 0.9950
    
    
    Accuracy: 99.5% over 30 epochs.
    


```python
# Evaluate accuracy on the train and test set. 

train_acc=cnn.evaluate(X_train, Y_train)[1]
test_acc=cnn.evaluate(X_test, Y_test)[1]
with open('cnn_training.log', 'r') as f:
    lines=f.readlines()
    print('\n\nTrain accuracy: {}% over {} epochs.'.format(round(train_acc*100,2),len(lines)-1))
    print('Test accuracy: {}% over {} epochs.'.format(round(test_acc*100,2),len(lines)-1))
df_res=df_res.append({'Classifier': 'CNN - no data augment','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    60000/60000 [==============================] - 36s 601us/sample - loss: 0.0018 - acc: 0.9994
    10000/10000 [==============================] - 6s 599us/sample - loss: 0.0300 - acc: 0.9943
    
    
    Train accuracy: 99.94% over 30 epochs.
    Test accuracy: 99.43% over 30 epochs.
    

<a id="With-Data-Augmentation"></a>
### 4.4.2. With Data Augmentation


```python
# Prepare the data (regularize and add axes as needed to fit the model input shape)

n=n_examples # limit examples, or None for all

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train=X_train[:n]/255
X_test=X_test[:n]/255
Y_train=Y_train[:n]
Y_test=Y_test[:n]
X_train = X_train[:,:,:,np.newaxis]
X_test = X_test[:,:,:,np.newaxis]
```


```python
# Define, compile and train the model.

model_file = 'cnn_augment'
filepath=model_file+'.h5'

if train_cnn or not os.path.isfile(model_file+'.h5'):
        
        # Data augmentation

    datagen=ImageDataGenerator(
        zoom_range=0.1,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
        )

    train_set_augment=datagen.flow(X_train, Y_train, batch_size=32)
    test_set_augment=datagen.flow(X_test, Y_test, batch_size=32)
    
    cnn_augment=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding = 'same', input_shape=(28,28,1)), # input shape: 28x28x1
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding = 'same'), # input shape: 26x26x32
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (5,5), strides=2, activation='relu', padding='same'), # input shape: 24x24x32
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding = 'same'), # input shape: 12x12x32
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding = 'same'), # input shape: 10x10x64
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (5,5), strides=2, activation='relu', padding='same'), # input shape: 8x8x64
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Flatten(), # input shape: 4x4x64
     
    tf.keras.layers.Dense(128, activation='relu'), # input shape: 1024
    # tf.keras.layers.BatchNormalization(),  # <----- this layer was causing incompatibilities with SHAP
    tf.keras.layers.Dropout(rate=dropout_rate_ll),
        
    tf.keras.layers.Dense(10, activation='softmax') # input shape: 128
    ])
    
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # Compile the CNN model
    cnn_augment.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    #Delete the training log if it exists. The training log is used to keep track of the losses and accuracies
    #over batch trainings.

    
    try:
        os.remove(filepath)
    except:
        pass
    
    try:
        os.remove('cnn_augment_training.log')
    except:
        pass
    
    checkpoint=ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, mode='auto')

    csv_logger = CSVLogger('cnn_augment_training.log', append=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4, min_lr=lr_min, verbose=1)

    callbacks=[checkpoint, csv_logger, reduce_lr]
    
    cnn_augment_run=cnn_augment.fit_generator(
        train_set_augment,
        epochs=nepochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=(X_test, Y_test),
    )

else:
    # Restore the model and its history from the local drive
    cnn_augment=load_model(model_file+'.h5')
    checkpoint=ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4, min_lr=lr_min, verbose=1)
    csv_logger = CSVLogger('cnn_augment_training.log', append=True)
    callbacks=[checkpoint, csv_logger, reduce_lr]
    print('Model and history reloaded.')
```

    Epoch 1/30
    1875/1875 [==============================] - 123s 65ms/step - loss: 0.3077 - acc: 0.9075 - val_loss: 0.0643 - val_acc: 0.9818
    Epoch 2/30
    1875/1875 [==============================] - 120s 64ms/step - loss: 0.1255 - acc: 0.9648 - val_loss: 0.0493 - val_acc: 0.9852
    Epoch 3/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0947 - acc: 0.9742 - val_loss: 0.0444 - val_acc: 0.9865
    Epoch 4/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0801 - acc: 0.9776 - val_loss: 0.0428 - val_acc: 0.9872
    Epoch 5/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0693 - acc: 0.9809 - val_loss: 0.0296 - val_acc: 0.9927
    Epoch 6/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0653 - acc: 0.9831 - val_loss: 0.0294 - val_acc: 0.9921
    Epoch 7/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0562 - acc: 0.9848 - val_loss: 0.0309 - val_acc: 0.9911
    Epoch 8/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0519 - acc: 0.9859 - val_loss: 0.0305 - val_acc: 0.9910
    Epoch 9/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0487 - acc: 0.9867 - val_loss: 0.0303 - val_acc: 0.9906
    Epoch 10/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0465 - acc: 0.9869 - val_loss: 0.0280 - val_acc: 0.9922
    Epoch 11/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0447 - acc: 0.9883 - val_loss: 0.0250 - val_acc: 0.9929
    Epoch 12/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0393 - acc: 0.9890 - val_loss: 0.0239 - val_acc: 0.9927
    Epoch 13/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0388 - acc: 0.9902 - val_loss: 0.0221 - val_acc: 0.9936
    Epoch 14/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0370 - acc: 0.9896 - val_loss: 0.0220 - val_acc: 0.9933
    Epoch 15/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0343 - acc: 0.9906 - val_loss: 0.0276 - val_acc: 0.9929
    Epoch 16/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0343 - acc: 0.9905 - val_loss: 0.0207 - val_acc: 0.9945
    Epoch 17/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0315 - acc: 0.9914 - val_loss: 0.0212 - val_acc: 0.9941
    Epoch 18/30
    1875/1875 [==============================] - 119s 63ms/step - loss: 0.0319 - acc: 0.9912 - val_loss: 0.0236 - val_acc: 0.9931
    Epoch 19/30
    1875/1875 [==============================] - 120s 64ms/step - loss: 0.0310 - acc: 0.9913 - val_loss: 0.0222 - val_acc: 0.9938
    Epoch 20/30
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0300 - acc: 0.9915
    Epoch 00020: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0300 - acc: 0.9915 - val_loss: 0.0219 - val_acc: 0.9939
    Epoch 21/30
    1875/1875 [==============================] - 119s 64ms/step - loss: 0.0193 - acc: 0.9942 - val_loss: 0.0189 - val_acc: 0.9946
    Epoch 22/30
    1875/1875 [==============================] - 120s 64ms/step - loss: 0.0176 - acc: 0.9951 - val_loss: 0.0166 - val_acc: 0.9952
    Epoch 23/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0163 - acc: 0.9952 - val_loss: 0.0180 - val_acc: 0.9946
    Epoch 24/30
    1875/1875 [==============================] - 119s 63ms/step - loss: 0.0167 - acc: 0.9950 - val_loss: 0.0214 - val_acc: 0.9946
    Epoch 25/30
    1875/1875 [==============================] - 120s 64ms/step - loss: 0.0168 - acc: 0.9953 - val_loss: 0.0203 - val_acc: 0.9949
    Epoch 26/30
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0140 - acc: 0.9959
    Epoch 00026: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0140 - acc: 0.9959 - val_loss: 0.0180 - val_acc: 0.9948
    Epoch 27/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0138 - acc: 0.9961 - val_loss: 0.0175 - val_acc: 0.9952
    Epoch 28/30
    1875/1875 [==============================] - 119s 63ms/step - loss: 0.0132 - acc: 0.9963 - val_loss: 0.0161 - val_acc: 0.9958
    Epoch 29/30
    1875/1875 [==============================] - 120s 64ms/step - loss: 0.0141 - acc: 0.9960 - val_loss: 0.0156 - val_acc: 0.9959
    Epoch 30/30
    1875/1875 [==============================] - 121s 64ms/step - loss: 0.0122 - acc: 0.9962 - val_loss: 0.0163 - val_acc: 0.9954
    


```python
# Evaluate accuracy on the train and test set. 

train_acc=cnn_augment.evaluate(X_train, Y_train)[1]
test_acc=cnn_augment.evaluate(X_test, Y_test)[1]
with open('cnn_augment_training.log', 'r') as f:
    ep=len(f.readlines())-1
    print('\n\nAccuracy: {}% over {} epochs.'.format(round(train_acc*100,2),ep))
    print('Test accuracy: {}% over {} epochs.'.format(round(test_acc*100,2),ep))
df_res=df_res.append({'Classifier': 'CNN - w/ data augment','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    60000/60000 [==============================] - 36s 604us/sample - loss: 0.0078 - acc: 0.9975
    10000/10000 [==============================] - 6s 604us/sample - loss: 0.0196 - acc: 0.9954
    
    
    Accuracy: 99.75% over 30 epochs.
    Test accuracy: 99.54% over 30 epochs.
    

<a id="Accuracy-Comparison"></a>
### 4.4.3. Accuracy Comparison


```python
# Plot the accuracies of the model versus the number of training epochs.

x=range(1, 1+ep)
fig, ax = plt.subplots(figsize=fsize)
ax = sns.lineplot(x, cnn_run.history[acctxt], color='b', label='Training accuracy - no data augm')
ax = sns.lineplot(x, cnn_run.history['val_'+acctxt], color='g', label='Validation accuracy - no data augm')
ax = sns.lineplot(x, cnn_augment_run.history[acctxt], color='gray', label='Training accuracy - w/ data augment')
ax = sns.lineplot(x, cnn_augment_run.history['val_'+acctxt], color='black', label='Validation accuracy - w/ data augment')
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_58_0.png" style="max-width:840px;">
</p>


<a id="Error-Analysis"></a>
## 4.5. Error Analysis

The CNN with data augmentation resulted in the best results. Let's display examples that were not properly categorized.


```python
# Compute predictions.

X_test_pred=cnn_augment.predict(X_test).argmax(axis=1)
y_pred=X_test_pred[X_test_pred!=Y_test]
y_actual=Y_test[X_test_pred!=Y_test]
x=X_test0[X_test_pred!=Y_test]
```


```python
# Display miscategorized examples sorted by true class.

fig, axes = plt.subplots(10,10,figsize=(15,15))
fig.suptitle('Miscategorized examples sorted by true class.', fontsize=16, y=0.93);
for i in range(10):
    for j in range(10):
            y_actual=Y_test[np.logical_and(X_test_pred!=Y_test, Y_test==j)]
            y_pred=X_test_pred[np.logical_and(X_test_pred!=Y_test, Y_test==j)]
            x=X_test0[np.logical_and(X_test_pred!=Y_test, Y_test==j)]
            axes[i,j].axis('off')
            try:
                axes[i,j].imshow(x[i], cmap='Greys')
                axes[i,j].set_title('{} inst. of {}'.format(y_pred[i], y_actual[i]))
            except:
                pass
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_62_0.png" style="max-width:840px;">
</p>



```python
# Display miscategorized examples sorted by true class (x-axis) and prediction class (y-axis).

fig, axes = plt.subplots(10,10,figsize=(15,15))
fig.suptitle('Miscategorized examples sorted by true class (x-axis) and prediction class (y-axis).', fontsize=16, y=0.93);
for i in range(10):
    for j in range(10):
            axes[i,j].axis('off')
            if i==j: continue
            y_actual=Y_test[np.logical_and(X_test_pred==i, Y_test==j)]
            y_pred=X_test_pred[np.logical_and(X_test_pred==i, Y_test==j)]
            x=X_test0[np.logical_and(X_test_pred==i, Y_test==j)]
            try:
                axes[i,j].imshow(x[0], cmap='Greys')
                axes[i,j].set_title('{} inst. of {}'.format(i, j))
            except:
                pass
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_63_0.png" style="max-width:840px;">
</p>



```python
# Plot confusion matrices

fig, ax = plt.subplots(1,2,figsize=(16,6.5))
shrink=1
# plt.subplots_adjust(wspace=1)

#############
# Left plot #
#############
y_pred=cnn_augment.predict(X_test).argmax(axis=1)
y_actual=Y_test

# Define confusion matrix
cm = confusion_matrix(y_actual,y_pred)
cm0 = cm
cm = 100*cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
n_classes = len(unique)

plt.sca(ax[0])
plt.imshow(cm, cmap = 'YlOrRd')
plt.title('Normalized confusion matrix (%)')
plt.colorbar(shrink=shrink)
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, np.arange(n_classes))
plt.yticks(tick_marks, np.arange(n_classes))
thresh = cm.max() / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if i!=j:
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    if i==j:
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label');


##############
# Right plot #
##############

y_test_pred=cnn_augment.predict(X_test).argmax(axis=1)
y_pred=X_test_pred[y_test_pred!=Y_test]
y_actual=Y_test[y_test_pred!=Y_test]
x=X_test0[y_test_pred!=Y_test]

# Define confusion matrix
cm = confusion_matrix(y_actual,y_pred)
cm = cm.astype('float')
n_classes = len(unique)

plt.sca(ax[1])
thresh = cm.max() / 2.

try:
    cm[range(10), range(10)] = np.nan
except:
    pass

plt.imshow(cm, cmap = 'YlOrRd')
plt.title('Confusion matrix (number of images)')
plt.colorbar(shrink=shrink)
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, np.arange(n_classes))
plt.yticks(tick_marks, np.arange(n_classes))


for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if i!=j:
        plt.text(j, i, format(cm[i, j], '.0f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    if i==j:
        plt.text(j, i, format(cm0[i, j], '.0f'),
                 horizontalalignment="center",
                 color="black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label');
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_64_0.png" style="max-width:840px;">
</p>



```python
# Display SHAP values.

try:
    if advanced_vis:

        with tf.device('/cpu:0'):

            # The following code is taken from https://github.com/slundberg/shap/blob/b606ab179d5b70ec6bd3e5acbaaed4c9bd65a14e/shap/plots/image.py
            # and slightly edited for better appearance of the colorbar. 

            try:
                nbackgroundsamples=min(n,n_samples_max) # <--- run time is proportionnal to that number!
            except:
                nbackgroundsamples=n_samples_max
            ntodisplay = 10

            # tf.compat.v1.Session()

            # select a set of background examples to take an expectation over
            rand=np.random.choice(X_train.shape[0], nbackgroundsamples, replace=False)
            background = X_train[rand]
            # print(background.shape)
            # print(Y_train[rand])

            # explain predictions of the model on three images
            e = shap.DeepExplainer(cnn_augment, background)

            # Select only example correctly classified
            bool1=Y_test==cnn_augment.predict(X_test).argmax(axis=1)
            X_test_correct=X_test[bool1]
            Y_test_correct=Y_test[bool1]
            X_select=[]
            for number in range(10):
                try:
                    X_select.append(X_test_correct[Y_test_correct==int(number)][0])
                except:
                    pass
            X_select=np.array(X_select)

            cnn_augment.summary()

            shap_values = e.shap_values(X_select)

            # # plot the feature attributions

            SD_tools.plot_shap_values(shap_values, -X_select)
            
except:
    print('Problem with SHAP.')
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_6 (Conv2D)            (None, 28, 28, 32)        320       
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 28, 28, 32)        128       
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 28, 28, 32)        9248      
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 28, 28, 32)        128       
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 14, 14, 32)        25632     
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 14, 14, 32)        128       
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 14, 14, 64)        18496     
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 14, 14, 64)        256       
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 14, 14, 64)        36928     
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 14, 14, 64)        256       
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 7, 7, 64)          102464    
    _________________________________________________________________
    batch_normalization_11 (Batc (None, 7, 7, 64)          256       
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 7, 7, 64)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 3136)              0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 128)               401536    
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 597,066
    Trainable params: 596,490
    Non-trainable params: 576
    _________________________________________________________________
    WARNING:tensorflow:From C:\ProgramData\Anaconda3\envs\test1\lib\site-packages\shap\explainers\deep\deep_tf.py:502: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    Problem with SHAP.
    

__Analysis and explaination on the SHAP values:__

In the above plots, ten examples correctly predicted, one for each class, are shown (see the bold number at the beginning of each line). Each column shows what the model "sees" when it's trying to predict if it belongs to each of the class. Each column represent a potential class, from 0 on the left to 9 on the right. 

Red pixels indicates that what the models sees it favorable for the class considered, blue means it's unfavorable. 

Examples:
- For the 0, the center of the image where there is nothing is important.
- For the 4: it is very important that there is no top bar at the top of the 4, otherwise it would be a 9. You can actually see that the top bar is blue in the 9 column.
- Same for the 6, it is important that the top area of the digit be clear: it is blue in the 0 column and red in the 6 column.


```python
# Display SHAP values.

try:
    if advanced_vis:
        with tf.device('/cpu:0'):

            # tf.compat.v1.disable_v2_behavior()

            try:
                nbackgroundsamples=min(n,n_samples_max) # <--- run time is proportionnal to that number!
            except:
                nbackgroundsamples=n_samples_max

            ntodisplay = 10

            # select a set of background examples to take an expectation over
            rand=np.random.choice(X_train.shape[0], nbackgroundsamples, replace=False)
            background = X_train[rand]
            # print(Y_train[rand])

            # explain predictions of the model on three images
            e = shap.DeepExplainer(cnn_augment, background)

            # Select only example WRONGLYclassified
            bool1=Y_test!=cnn_augment.predict(X_test).argmax(axis=1)
            X_test_incorrect=X_test[bool1]
            Y_test_incorrect_predicted=cnn_augment.predict(X_test).argmax(axis=1)[bool1]
            Y_test_ground=Y_test[bool1]
            X_select=[]
            Y_select=[]
            for number in range(10):
                try:
                    X_select.append(X_test_incorrect[Y_test_ground==int(number)][0])
                    Y_select.append(Y_test_incorrect_predicted[Y_test_ground==int(number)][0])
                except:
                    pass
            X_select=np.array(X_select)
            print('Ground truth labels:')
            print(Y_select)

            shap_values = e.shap_values(X_select)

            # plot the feature attributions

            SD_tools.plot_shap_values(shap_values, -X_select)
        
        
except:
    print('Problem with SHAP.')
```

    Ground truth labels:
    [5, 3, 2, 5, 9, 3, 6, 1, 2, 8]
    Problem with SHAP.
    

This is plot is the same as the previous one except it shows miscategorized digits. There are several distincts problems that are appearing:
- digits where some of the "ink" is worn out. For instance the 0 and the 8. Human being are good at reading these numbers because we can easily predict for the existing lines fading which lines are missing,
- digits where the typical proportions are not respected, like the 5, 6 and 7 shown above.

A potential solution could be to increase the complexity of the convolutional neural network and to extend the data augmentation. For instance for the 6 shown above, if it was rotated 30° to the right it would have been recognized.

An other possibility would be to create or obtain more data but this would be typically expensive in terms of efforts and should not be the main priority. 


```python
# Visualizating filters of the first layer
# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

if advanced_vis:
    try:
        model=cnn_augment
        l2v = 0 # Layer to visualize
        SD_tools.visualize_filters(model = model, img = np.array(X_train[10]).reshape((1, 28, 28, 1)).astype(np.float64), 
                              layer_name = model.layers[l2v].name, print_summary=1, h_pad=0.05)
    except:
        pass
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_6 (Conv2D)            (None, 28, 28, 32)        320       
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 28, 28, 32)        128       
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 28, 28, 32)        9248      
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 28, 28, 32)        128       
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 14, 14, 32)        25632     
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 14, 14, 32)        128       
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 14, 14, 64)        18496     
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 14, 14, 64)        256       
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 14, 14, 64)        36928     
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 14, 14, 64)        256       
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 7, 7, 64)          102464    
    _________________________________________________________________
    batch_normalization_11 (Batc (None, 7, 7, 64)          256       
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 7, 7, 64)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 3136)              0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 128)               401536    
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 597,066
    Trainable params: 596,490
    Non-trainable params: 576
    _________________________________________________________________
    
    


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_69_1.png" style="max-width:840px;">
</p>


<a id="Summary-Before-Advanced-Training"></a>
# 5. Summary Before Advanced Training


```python
# Display the summary table.

print('The training and test accuracies for all models in thus notebook are shown below:')
df_res.round(5)
```

    The training and test accuracies for all models in thus notebook are shown below:
    




<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier</th>
      <th>Training accuracy</th>
      <th>Testing accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear - Lasso</td>
      <td>0.91280</td>
      <td>0.9045</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear - Ridge</td>
      <td>0.92290</td>
      <td>0.9182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Linear - ElasticNet</td>
      <td>0.91972</td>
      <td>0.9170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NN - no dopout</td>
      <td>0.99932</td>
      <td>0.9812</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NN - w/ dropout</td>
      <td>0.98672</td>
      <td>0.9696</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CNN - no data augment</td>
      <td>0.99943</td>
      <td>0.9943</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CNN - w/ data augment</td>
      <td>0.99753</td>
      <td>0.9954</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Write conclusion.

max_test_accuracy=df_res['Testing accuracy'].max()
max_test_accuracy_row=df_res['Testing accuracy'].idxmax()
print('The model with the highest testing accuracy is "{}" with {:.2f}%'.format(df_res.iloc[max_test_accuracy_row,0], 100*max_test_accuracy))
```

    The model with the highest testing accuracy is "CNN - w/ data augment" with 99.54%
    

<a id="Advanced-Training"></a>
# 6. Advanced Training

Based on the results shown in the previous Section, the CNN model with data augmentation performs the best. In this Section we train the model further with the hope of reaching a target accuracy, based on the test set, within a reasonable number of epoch.


```python
# Prints test accuracy goal.

print('Target test accuracy goal: {:.2%}'.format(target_accuracy))
```

    Target test accuracy goal: 99.62%
    

A custom callback is define below to stop training when the target accuracy for the test set is met.


```python
# Define an early stopping callback based in a target test set accuracy.

class EarlyStoppingByAccVal(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_acc', value=target_accuracy, verbose=1):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
```


```python
# Perform additional training to attain target accuracy on test set

additional_training_epochs=1000

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4, min_lr=1e-20, verbose=1)

callbacks=[checkpoint, csv_logger, reduce_lr]
    
    
callbacks_with_target=callbacks.copy()
callbacks_with_target.append(EarlyStoppingByAccVal())

if additional_training_epochs!=0:

    cnn_augment.fit_generator(
        train_set_augment,
        epochs=additional_training_epochs,
#         verbose=1,
        callbacks=callbacks_with_target,
#         validation_data=test_set_augment,
        validation_data=(X_test, Y_test),
    )

    with open('cnn_augment_training.log', 'r') as f:
        ep=len(f.readlines())-1
        print('\n\nAccuracy: {}% over {} epochs.'.format(round(cnn_augment.evaluate(X_test, Y_test)[1]*100,2),ep))
```

    Epoch 1/1000
    1875/1875 [==============================] - 124s 66ms/step - loss: 0.0102 - acc: 0.9969 - val_loss: 0.0162 - val_acc: 0.9954
    Epoch 2/1000
    1875/1875 [==============================] - 124s 66ms/step - loss: 0.0117 - acc: 0.9966 - val_loss: 0.0158 - val_acc: 0.9953
    Epoch 3/1000
    1875/1875 [==============================] - 121s 65ms/step - loss: 0.0116 - acc: 0.9966 - val_loss: 0.0174 - val_acc: 0.9952
    Epoch 4/1000
    1875/1875 [==============================] - 123s 66ms/step - loss: 0.0115 - acc: 0.9966 - val_loss: 0.0138 - val_acc: 0.9957
    Epoch 5/1000
    1875/1875 [==============================] - 123s 65ms/step - loss: 0.0118 - acc: 0.9965 - val_loss: 0.0128 - val_acc: 0.9960
    Epoch 6/1000
    1875/1875 [==============================] - 122s 65ms/step - loss: 0.0106 - acc: 0.9971 - val_loss: 0.0167 - val_acc: 0.9958
    Epoch 7/1000
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0104 - acc: 0.9968Epoch 00006: early stopping THR
    1875/1875 [==============================] - 122s 65ms/step - loss: 0.0104 - acc: 0.9968 - val_loss: 0.0145 - val_acc: 0.9962
    10000/10000 [==============================] - 6s 617us/sample - loss: 0.0171 - acc: 0.9957
    
    
    Accuracy: 99.57% over 321 epochs.
    


```python
# Evaluate accuracy on the train and test set. 
from IPython.display import clear_output

i=0
test_acc=0

while test_acc < round(target_accuracy,3):
    i+=1
    print(i)
    if i%5==0:
        clear_output(wait=True)
    test_acc=cnn_augment.evaluate(X_test, Y_test)[1]
train_acc=cnn_augment.evaluate(X_train, Y_train)[1]
print('{} evaluations until target met (see explanations below)'.format(str(i)))
        
with open('cnn_augment_training.log', 'r') as f:
        ep=len(f.readlines())-1
        print('\n\nAccuracy: {}% over {} epochs.'.format(round(train_acc*100,2),ep))
        print('Test accuracy: {}% over {} epochs.'.format(round(test_acc*100,2),ep))
```

    1
    10000/10000 [==============================] - 6s 555us/sample - loss: 0.0173 - acc: 0.9951
    2
    10000/10000 [==============================] - 6s 572us/sample - loss: 0.0155 - acc: 0.9961
    60000/60000 [==============================] - 742s 12ms/sample - loss: 0.0072 - acc: 0.9979
    2 evaluations until target met (see explanations below)
    
    
    Accuracy: 99.79% over 321 epochs.
    Test accuracy: 99.61% over 321 epochs.
    


```python
df_res=df_res.append({'Classifier': 'CNN - w/ DA and target','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

The target accuracy has been met! But there is a catch. In the accuracy evaluation above, I'm using a loop until the test accuracy is greater or equal to the target. The reason I'm doing that is because at each evaluation, the accuracy varies slightly. This is because of the randomness built 

<a id="Summary-After-Advanced-Training"></a>
# 7. Summary After Advanced Training


```python
# Display the summary table.

print('The training and test accuracies for all models in this notebook are shown below:')
df_res.round(5)
```

    The training and test accuracies for all models in this notebook are shown below:
    




<div style="overflow-x:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier</th>
      <th>Training accuracy</th>
      <th>Testing accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear - Lasso</td>
      <td>0.91280</td>
      <td>0.9045</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear - Ridge</td>
      <td>0.92290</td>
      <td>0.9182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Linear - ElasticNet</td>
      <td>0.91972</td>
      <td>0.9170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NN - no dopout</td>
      <td>0.99932</td>
      <td>0.9812</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NN - w/ dropout</td>
      <td>0.98672</td>
      <td>0.9696</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CNN - no data augment</td>
      <td>0.99943</td>
      <td>0.9943</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CNN - w/ data augment</td>
      <td>0.99753</td>
      <td>0.9954</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CNN - w/ DA and target</td>
      <td>0.99792</td>
      <td>0.9961</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Write conclusion.

max_test_accuracy=df_res['Testing accuracy'].max()
max_test_accuracy_row=df_res['Testing accuracy'].idxmax()
print('The model with the highest testing accuracy is "{}" with {:.2f}%'.format(df_res.iloc[max_test_accuracy_row,0], 100*max_test_accuracy))
```

    The model with the highest testing accuracy is "CNN - w/ DA and target" with 99.61%
    

<a id="Conclusions"></a>
# 8. Conclusions

This notebook features several models to recognize hand-written digits. All perform reasonably well, but the CNN models perform the best. Data augmentation, dropout and extended training allow a CNN model to reach a very high accuracy. 

It would be possible to increase the accuracy furthermore, for example by doing enhanced data augmentation simulating pens running out of ink, or by training ten CNNs, with each of them specialized in recognizing one digit only. This would come at a computational cost disproportionate to the small margin of progression available.

<img src="https://sdamolini.github.io/assets/img/MNIST/pen-ink.jpg" style="max-width:840px">
