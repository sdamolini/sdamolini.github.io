---
layout: post
title: "Handwritten Digit Recognition (MNIST)"
img:     MNIST_thumbnail.jpg
date: 2020-02-28 12:54:00 +0300
description: None. 
tag: ['MNIST', 'CNN', 'SHAP']
---
<a id="Introduction"></a>
# 1. Introduction

This is a take on the well know MNIST dataset (Modified National Institute of Standards and Technology) that is comprised of handritten digits and it commonly used to practice recognition algorithms.

<a id="Imports"></a>
# 2. Imports

<a id="Modules-and-Parameters"></a>
## 2.1. Modules and Parameters


```python
# Import personnal set of tools
importlib.reload(SD_tools)
import SD.tools as SD_tools
from SD.tools import package_version as v
```

    -----------------------------------------------
    SD_tools version 0.2.1 were succesfully loaded.       
    -----------------------------------------------
    


```python
import pandas as pd
v(pd)

import numpy as np
v(np)

import os

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

from tensorflow import keras
from keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
v(keras)

import pickle

import itertools

import shap
from shap.plots import colors 
v(shap)

import importlib
```

    Pandas version: 0.25.3
    Numpy version: 1.18.1
    Matplotlib version: 3.1.2
    Sklearn version: 0.22.1
    Tensorflow version: 1.15.0
    Tensorflow.python.keras.api._v1.keras version: 2.2.4-tf
    Shap version: 0.34.0
    


```python
# Define custom parameters

fsize=(15,5) # Figure size for plots
n_examples=None # limit examples for training, or None for all
nepochs=20 # Number of epochs for NN and CNN
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
print('The train set contains {} images of {}x{} pixels; the test set contains {} images of {}x{} pixels.'.format(
    *(X_train.shape), *(X_test.shape)))
print('There are {} unique values in the set.'.format(len(np.unique(Y_train))))
```

    The train set contains 60000 images of 28x28 pixels; the test set contains 10000 images of 28x28 pixels.
    There are 10 unique values in the set.
    


```python
unique = np.unique(Y_train); unique
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)



Let's display some samples from the train set.


```python
fig, axes = plt.subplots(10,10,figsize=(8,8))
for digit in unique:
    digit_images=X_train[Y_train==digit][20:]
    for k in range(10):
        axes[k,digit].imshow(digit_images[k], cmap='Greys')
        axes[k,digit].axis('off')
plt.suptitle('Samples from the dataset.', y=0.92);
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_14_0.png" style="max-width:840px;">
</p>


Let's check the distribution of the unique classes in the training set.


```python
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
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_16_0.png" style="max-width:840px;">
</p>


Conclusion: the classes in the train and test sets are well distributed and can be used as is.

<a id="Modeling"></a>
# 4. Modeling

Various models will be trained to recognized these handwritten digits:
    - Logistic regression models with various regularizations (Lasso, Ridge, Ellastic net)
    - Neural networks with or without regularization (dropout)
    - Convolutional Neural Networks with or without data augmentation

<a id="Normalization"></a>
## 4.1. Normalization

Data is normalized so pixel intensities are between 0 and 1.


```python
n=n_examples # limit examples, or None for all

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train=X_train[:n]/255
X_test=X_test[:n]/255
Y_train=Y_train[:n]
Y_test=Y_test[:n]
```


```python
X_train0 = X_train
X_train = X_train.reshape(X_train.shape[0],-1)
X_test0 = X_test
X_test = X_test.reshape(X_test.shape[0],-1)
```


```python
X_train.shape, Y_train.shape
```




    ((60000, 784), (60000,))



<a id="Logistic-Regression"></a>
## 4.2. Logistic Regression


```python
df_res = pd.DataFrame(columns=['Classifier','Training accuracy','Testing accuracy'])
```


```python
clf=sklearn.linear_model.SGDClassifier(penalty='l1')
clf.fit(X_train,Y_train)
train_acc=accuracy_score(clf.predict(X_train), Y_train)
test_acc=accuracy_score(clf.predict(X_test), Y_test)
print('Train accuracy: {}, test accuracy: {}.'.format(round(train_acc,4),round(test_acc,4)))
df_res=df_res.append({'Classifier': 'Linear - Lasso','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True);
```

    Train accuracy: 0.9144, test accuracy: 0.9046.
    


```python
clf=sklearn.linear_model.SGDClassifier(penalty='l2')
clf.fit(X_train,Y_train)
train_acc=accuracy_score(clf.predict(X_train), Y_train)
test_acc=accuracy_score(clf.predict(X_test), Y_test)
print('Train accuracy: {}, test accuracy: {}.'.format(round(train_acc,4),round(test_acc,4)))
df_res=df_res.append({'Classifier': 'Linear - Ridge','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True);
```

    Train accuracy: 0.9229, test accuracy: 0.9175.
    


```python
clf=sklearn.linear_model.SGDClassifier(penalty='elasticnet')
clf.fit(X_train,Y_train)
train_acc=accuracy_score(clf.predict(X_train), Y_train)
test_acc=accuracy_score(clf.predict(X_test), Y_test)
print('Train accuracy: {}, test accuracy: {}.'.format(round(train_acc,4),round(test_acc,4)))
df_res=df_res.append({'Classifier': 'Linear - ElasticNet','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    Train accuracy: 0.9202, test accuracy: 0.9147.
    


```python
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
      <td>0.9144</td>
      <td>0.9046</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear - Ridge</td>
      <td>0.9229</td>
      <td>0.9175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Linear - ElasticNet</td>
      <td>0.9202</td>
      <td>0.9147</td>
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
    
On the model incorporating regularization, a 30% dropout layer is added after each hidden layer.

The models are optimized using Adam, using a cross entropy loss function and accuracy as the metrics for performance. Models are fitted over 50 epochs.

<a id="No-Dropout"></a>
### 4.3.1. No Dropout


```python
# Build the model leayers

nn_nodrop = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu',),
    tf.keras.layers.Dense(128, activation='relu',),
    tf.keras.layers.Dense(64, activation='relu',),
    tf.keras.layers.Dense(32, activation='relu',),
    tf.keras.layers.Dense(10, activation='softmax',),
])

# Compile the model
nn_nodrop.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['acc'],)

# Train the model
nn_nodrop_run=nn_nodrop.fit(X_train,
              Y_train, 
              epochs=nepochs,
             validation_data=(X_test, Y_test),
             )
```

    WARNING:tensorflow:From C:\ProgramData\Anaconda3\lib\site-packages\tensorflow_core\python\ops\resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 8s 135us/sample - loss: 0.2349 - acc: 0.9302 - val_loss: 0.1433 - val_acc: 0.9559
    Epoch 2/20
    60000/60000 [==============================] - 8s 130us/sample - loss: 0.0975 - acc: 0.9700 - val_loss: 0.0882 - val_acc: 0.9726
    Epoch 3/20
    60000/60000 [==============================] - 8s 129us/sample - loss: 0.0684 - acc: 0.9795 - val_loss: 0.0823 - val_acc: 0.9752
    Epoch 4/20
    60000/60000 [==============================] - 7s 124us/sample - loss: 0.0541 - acc: 0.9834 - val_loss: 0.0828 - val_acc: 0.9770
    Epoch 5/20
    60000/60000 [==============================] - 7s 120us/sample - loss: 0.0449 - acc: 0.9860 - val_loss: 0.0814 - val_acc: 0.9773
    Epoch 6/20
    60000/60000 [==============================] - 7s 123us/sample - loss: 0.0389 - acc: 0.9875 - val_loss: 0.0762 - val_acc: 0.9796
    Epoch 7/20
    60000/60000 [==============================] - 8s 128us/sample - loss: 0.0326 - acc: 0.9899 - val_loss: 0.0889 - val_acc: 0.9789
    Epoch 8/20
    60000/60000 [==============================] - 7s 121us/sample - loss: 0.0254 - acc: 0.9919 - val_loss: 0.0900 - val_acc: 0.9769
    Epoch 9/20
    60000/60000 [==============================] - 9s 154us/sample - loss: 0.0253 - acc: 0.9920 - val_loss: 0.1088 - val_acc: 0.9725
    Epoch 10/20
    60000/60000 [==============================] - 8s 131us/sample - loss: 0.0243 - acc: 0.9923 - val_loss: 0.0835 - val_acc: 0.9790
    Epoch 11/20
    60000/60000 [==============================] - 8s 126us/sample - loss: 0.0199 - acc: 0.9935 - val_loss: 0.0851 - val_acc: 0.9807
    Epoch 12/20
    60000/60000 [==============================] - 7s 122us/sample - loss: 0.0192 - acc: 0.9941 - val_loss: 0.0843 - val_acc: 0.9796
    Epoch 13/20
    60000/60000 [==============================] - 7s 122us/sample - loss: 0.0168 - acc: 0.9948 - val_loss: 0.1242 - val_acc: 0.9754
    Epoch 14/20
    60000/60000 [==============================] - 7s 122us/sample - loss: 0.0168 - acc: 0.9951 - val_loss: 0.1050 - val_acc: 0.9789
    Epoch 15/20
    60000/60000 [==============================] - 8s 135us/sample - loss: 0.0163 - acc: 0.9949 - val_loss: 0.0824 - val_acc: 0.9809
    Epoch 16/20
    60000/60000 [==============================] - 7s 122us/sample - loss: 0.0128 - acc: 0.9961 - val_loss: 0.1142 - val_acc: 0.9775
    Epoch 17/20
    60000/60000 [==============================] - 9s 147us/sample - loss: 0.0138 - acc: 0.9960 - val_loss: 0.1009 - val_acc: 0.9795
    Epoch 18/20
    60000/60000 [==============================] - 10s 162us/sample - loss: 0.0121 - acc: 0.9966 - val_loss: 0.1051 - val_acc: 0.9805
    Epoch 19/20
    60000/60000 [==============================] - 7s 121us/sample - loss: 0.0141 - acc: 0.9961 - val_loss: 0.0803 - val_acc: 0.9815
    Epoch 20/20
    60000/60000 [==============================] - 7s 119us/sample - loss: 0.0109 - acc: 0.9971 - val_loss: 0.1144 - val_acc: 0.9796
    


```python
train_acc=nn_nodrop.evaluate(X_train, Y_train)[1]
test_acc=nn_nodrop.evaluate(X_test, Y_test)[1]
print('\n\nTrain accuracy: {}% over {} epochs.'.format(round(train_acc*100,2),len(nn_nodrop_run.history)))
print('Test accuracy: {}% over {} epochs.'.format(round(test_acc*100,2),len(nn_nodrop_run.history)))
df_res=df_res.append({'Classifier': 'NN - no dopout','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    60000/60000 [==============================] - 3s 48us/sample - loss: 0.0181 - acc: 0.9948
    10000/10000 [==============================] - 1s 52us/sample - loss: 0.1144 - acc: 0.9796
    
    
    Train accuracy: 99.48% over 4 epochs.
    Test accuracy: 97.96% over 4 epochs.
    


```python
nn_nodrop.history
```




    <tensorflow.python.keras.callbacks.History at 0x179800cb348>



<a id="With-Dropout"></a>
### 4.3.2. With Dropout


```python
# Build the model leayers

dropout_rate=0.30

nn_drop = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
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
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 10s 166us/sample - loss: 0.5251 - acc: 0.8437 - val_loss: 0.1526 - val_acc: 0.9573
    Epoch 2/20
    60000/60000 [==============================] - 9s 156us/sample - loss: 0.2404 - acc: 0.9402 - val_loss: 0.1318 - val_acc: 0.9666
    Epoch 3/20
    60000/60000 [==============================] - 9s 156us/sample - loss: 0.1954 - acc: 0.9519 - val_loss: 0.1115 - val_acc: 0.9704
    Epoch 4/20
    60000/60000 [==============================] - 9s 157us/sample - loss: 0.1629 - acc: 0.9608 - val_loss: 0.0990 - val_acc: 0.9735
    Epoch 5/20
    60000/60000 [==============================] - 10s 160us/sample - loss: 0.1484 - acc: 0.9639 - val_loss: 0.0969 - val_acc: 0.9748
    Epoch 6/20
    60000/60000 [==============================] - 10s 161us/sample - loss: 0.1313 - acc: 0.9664 - val_loss: 0.0936 - val_acc: 0.9758
    Epoch 7/20
    60000/60000 [==============================] - 10s 159us/sample - loss: 0.1197 - acc: 0.9694 - val_loss: 0.0920 - val_acc: 0.9763
    Epoch 8/20
    60000/60000 [==============================] - 10s 158us/sample - loss: 0.1152 - acc: 0.9707 - val_loss: 0.0861 - val_acc: 0.9784
    Epoch 9/20
    60000/60000 [==============================] - 10s 160us/sample - loss: 0.1044 - acc: 0.9735 - val_loss: 0.0877 - val_acc: 0.9781
    Epoch 10/20
    60000/60000 [==============================] - 10s 163us/sample - loss: 0.0996 - acc: 0.9748 - val_loss: 0.0915 - val_acc: 0.9794
    Epoch 11/20
    60000/60000 [==============================] - 10s 166us/sample - loss: 0.0968 - acc: 0.9750 - val_loss: 0.0941 - val_acc: 0.9781
    Epoch 12/20
    60000/60000 [==============================] - 10s 160us/sample - loss: 0.0902 - acc: 0.9773 - val_loss: 0.0922 - val_acc: 0.9788
    Epoch 13/20
    60000/60000 [==============================] - 10s 162us/sample - loss: 0.0879 - acc: 0.9774 - val_loss: 0.0793 - val_acc: 0.9814
    Epoch 14/20
    60000/60000 [==============================] - 10s 163us/sample - loss: 0.0819 - acc: 0.9794 - val_loss: 0.0837 - val_acc: 0.9796
    Epoch 15/20
    60000/60000 [==============================] - 10s 162us/sample - loss: 0.0801 - acc: 0.9799 - val_loss: 0.0847 - val_acc: 0.9800
    Epoch 16/20
    60000/60000 [==============================] - 10s 164us/sample - loss: 0.0760 - acc: 0.9814 - val_loss: 0.0967 - val_acc: 0.9797
    Epoch 17/20
    60000/60000 [==============================] - 10s 163us/sample - loss: 0.0726 - acc: 0.9817 - val_loss: 0.0919 - val_acc: 0.9803
    Epoch 18/20
    60000/60000 [==============================] - 10s 162us/sample - loss: 0.0736 - acc: 0.9817 - val_loss: 0.1072 - val_acc: 0.9807
    Epoch 19/20
    60000/60000 [==============================] - 10s 164us/sample - loss: 0.0676 - acc: 0.9828 - val_loss: 0.0895 - val_acc: 0.9811
    Epoch 20/20
    60000/60000 [==============================] - 10s 164us/sample - loss: 0.0706 - acc: 0.9819 - val_loss: 0.0835 - val_acc: 0.9812
    


```python
train_acc=nn_drop.evaluate(X_train, Y_train)[1]
test_acc=nn_drop.evaluate(X_test, Y_test)[1]
print('\n\nTrain accuracy: {}% over {} epochs.'.format(round(train_acc*100,2),len(nn_drop_run.history)))
print('Test accuracy: {}% over {} epochs.'.format(round(test_acc*100,2),len(nn_drop_run.history)))
df_res=df_res.append({'Classifier': 'NN - w/ dropout','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    60000/60000 [==============================] - 3s 54us/sample - loss: 0.0182 - acc: 0.9945
    10000/10000 [==============================] - 1s 55us/sample - loss: 0.0835 - acc: 0.9812
    
    
    Train accuracy: 99.45% over 4 epochs.
    Test accuracy: 98.12% over 4 epochs.
    

<a id="Accuracy-Comparison"></a>
### 4.3.3. Accuracy Comparison


```python
x=nn_nodrop_run.epoch
fig, ax = plt.subplots(figsize=fsize)
ax = sns.lineplot(x, nn_nodrop_run.history['acc'], color='b', label='Training accuracy - no dropout')
ax = sns.lineplot(x, nn_nodrop_run.history['val_acc'], color='g', label='Validation accuracy - no dropout')
ax = sns.lineplot(x, nn_drop_run.history['acc'], color='gray', label='Training accuracy - w/ dropout')
ax = sns.lineplot(x, nn_drop_run.history['val_acc'], color='black', label='Validation accuracy - w/ dropout')
plt.title('Training and validation accuracy for neural networks')
plt.ylabel('Accuracy')
plt.xlabel('Epochs');
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_41_0.png" style="max-width:840px;">
</p>


<a id="Error-Analysis"></a>
### 4.3.4. Error Analysis

The neural network with dropout resulted in the best results. Let's display examples that were not properly categorized.


```python
X_test_pred=nn_drop.predict(X_test).argmax(axis=1)
y_pred=X_test_pred[X_test_pred!=Y_test]
y_actual=Y_test[X_test_pred!=Y_test]
x=X_test0[X_test_pred!=Y_test]
```


```python
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
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_45_0.png" style="max-width:840px;">
</p>



```python
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
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_46_0.png" style="max-width:840px;">
</p>



```python
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
y_pred=X_test_pred[X_test_pred!=Y_test]
y_actual=Y_test[y_test_pred!=Y_test]
x=X_test0[X_test_pred!=Y_test]

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
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_47_0.png" style="max-width:840px;">
</p>


The matrix on the left shows the percentage of confusion between predicted labels and ground truth labels, in percentage. It can be seen that a large majority of predictions are correct.  

It is however, a little difficult to tell which classes are especially poorly predicted. On the matrix on the right, the true positives are removed from the set to make the mistakes more apparent, and the total number of cases in shown as opposed to the percentage. 

<a id="Convolutional-Neural-Network"></a>
## 4.4. Convolutional Neural Network

<a id="No-Data-Augmentation"></a>
### 4.4.1. No Data Augmentation


```python
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
cnn=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (7,7), activation='relu', input_shape=(28,28,1), padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Conv2D(48, (5,5), activation='relu', input_shape=(28,28,1), padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Flatten(),
     
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Dense(10, activation='softmax')
])
```


```python
cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```


```python
# cnn.summary()
```


```python
try:
    os.remove(filepath)
    os.remove('training.log')
except:
    pass
```


```python
filepath='cnn_no_aug.h5'
checkpoint=ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='auto')
```


```python
csv_logger = CSVLogger('training.log', append=True)
```


```python
callbacks=[checkpoint, csv_logger]
```


```python
cnn_run=cnn.fit(
    X_train,
    Y_train,
#     batch_size=128,
    epochs=nepochs,
    validation_data=(X_test, Y_test),
#     validation_steps=len(X_test)//32,
    callbacks=callbacks,
#     verbose=1
)

with open('training.log', 'r') as f:
    lines=f.readlines()
    print('\n\nAccuracy: {}% over {} epochs.'.format(round(cnn.evaluate(X_test, Y_test)[1]*100,2),len(lines)-1))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.1676 - acc: 0.9475
    Epoch 00001: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 165s 3ms/sample - loss: 0.1675 - acc: 0.9475 - val_loss: 0.0441 - val_acc: 0.9851
    Epoch 2/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0670 - acc: 0.9801
    Epoch 00002: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 162s 3ms/sample - loss: 0.0671 - acc: 0.9801 - val_loss: 0.0320 - val_acc: 0.9895
    Epoch 3/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0535 - acc: 0.9840
    Epoch 00003: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 162s 3ms/sample - loss: 0.0535 - acc: 0.9840 - val_loss: 0.0306 - val_acc: 0.9901
    Epoch 4/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0452 - acc: 0.9861
    Epoch 00004: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 161s 3ms/sample - loss: 0.0452 - acc: 0.9861 - val_loss: 0.0347 - val_acc: 0.9885
    Epoch 5/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0427 - acc: 0.9873
    Epoch 00005: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 161s 3ms/sample - loss: 0.0427 - acc: 0.9873 - val_loss: 0.0207 - val_acc: 0.9928
    Epoch 6/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0381 - acc: 0.9878
    Epoch 00006: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 160s 3ms/sample - loss: 0.0381 - acc: 0.9879 - val_loss: 0.0233 - val_acc: 0.9919
    Epoch 7/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0376 - acc: 0.9884
    Epoch 00007: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 160s 3ms/sample - loss: 0.0377 - acc: 0.9884 - val_loss: 0.0266 - val_acc: 0.9909
    Epoch 8/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0311 - acc: 0.9905
    Epoch 00008: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 162s 3ms/sample - loss: 0.0311 - acc: 0.9905 - val_loss: 0.0205 - val_acc: 0.9934
    Epoch 9/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0307 - acc: 0.9903
    Epoch 00009: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 160s 3ms/sample - loss: 0.0306 - acc: 0.9903 - val_loss: 0.0221 - val_acc: 0.9933
    Epoch 10/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0310 - acc: 0.9903
    Epoch 00010: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 160s 3ms/sample - loss: 0.0310 - acc: 0.9903 - val_loss: 0.0183 - val_acc: 0.9936
    Epoch 11/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0277 - acc: 0.9913
    Epoch 00011: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 159s 3ms/sample - loss: 0.0277 - acc: 0.9913 - val_loss: 0.0180 - val_acc: 0.9939
    Epoch 12/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0276 - acc: 0.9914
    Epoch 00012: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 160s 3ms/sample - loss: 0.0276 - acc: 0.9914 - val_loss: 0.0193 - val_acc: 0.9931
    Epoch 13/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0276 - acc: 0.9913
    Epoch 00013: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 160s 3ms/sample - loss: 0.0277 - acc: 0.9913 - val_loss: 0.0201 - val_acc: 0.9935
    Epoch 14/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0258 - acc: 0.9919
    Epoch 00014: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 159s 3ms/sample - loss: 0.0257 - acc: 0.9919 - val_loss: 0.0228 - val_acc: 0.9938
    Epoch 15/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0253 - acc: 0.9927
    Epoch 00015: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 161s 3ms/sample - loss: 0.0253 - acc: 0.9927 - val_loss: 0.0258 - val_acc: 0.9924
    Epoch 16/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0237 - acc: 0.9928
    Epoch 00016: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 161s 3ms/sample - loss: 0.0237 - acc: 0.9928 - val_loss: 0.0207 - val_acc: 0.9939
    Epoch 17/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0241 - acc: 0.9927
    Epoch 00017: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 162s 3ms/sample - loss: 0.0241 - acc: 0.9927 - val_loss: 0.0244 - val_acc: 0.9932
    Epoch 18/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0214 - acc: 0.9938
    Epoch 00018: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 161s 3ms/sample - loss: 0.0214 - acc: 0.9938 - val_loss: 0.0197 - val_acc: 0.9944
    Epoch 19/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0226 - acc: 0.9933
    Epoch 00019: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 161s 3ms/sample - loss: 0.0226 - acc: 0.9933 - val_loss: 0.0188 - val_acc: 0.9950
    Epoch 20/20
    59968/60000 [============================>.] - ETA: 0s - loss: 0.0212 - acc: 0.9933
    Epoch 00020: saving model to cnn_no_aug.h5
    60000/60000 [==============================] - 168s 3ms/sample - loss: 0.0212 - acc: 0.9933 - val_loss: 0.0204 - val_acc: 0.9941
    10000/10000 [==============================] - 9s 929us/sample - loss: 0.0204 - acc: 0.9941
    
    
    Accuracy: 99.41% over 120 epochs.
    


```python
train_acc=cnn.evaluate(X_train, Y_train)[1]
test_acc=cnn.evaluate(X_test, Y_test)[1]
print('\n\nTrain accuracy: {}% over {} epochs.'.format(round(train_acc*100,2),len(nn_nodrop_run.history)))
print('Test accuracy: {}% over {} epochs.'.format(round(test_acc*100,2),len(nn_nodrop_run.history)))
df_res=df_res.append({'Classifier': 'CNN - no data augment','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    60000/60000 [==============================] - 57s 942us/sample - loss: 0.0035 - acc: 0.9990
    10000/10000 [==============================] - 9s 948us/sample - loss: 0.0204 - acc: 0.9941
    
    
    Train accuracy: 99.9% over 4 epochs.
    Test accuracy: 99.41% over 4 epochs.
    


```python

```


```python

```


```python

```


```python

```


```python

```

<a id="With-Data-Augmentation"></a>
### 4.4.2. With Data Augmentation


```python
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
# Data augmentation

datagen=ImageDataGenerator(
    zoom_range=0.12,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.10
    )

train_set_augment=datagen.flow(X_train, Y_train, batch_size=32)
test_set_augment=datagen.flow(X_test, Y_test, batch_size=32)
```


```python
cnn_augment=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (7,7), activation='relu', input_shape=(28,28,1), padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Conv2D(48, (5,5), activation='relu', input_shape=(28,28,1), padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Flatten(),
     
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(rate=dropout_rate),
    
    tf.keras.layers.Dense(10, activation='softmax')
])

cnn_augment.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

cnn_augment_run=cnn_augment.fit_generator(
    train_set_augment,
    epochs=nepochs,
    verbose=1,
    validation_data=test_set_augment)
```

    Epoch 1/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.3370 - acc: 0.8938Epoch 1/20
    1875/1875 [==============================] - 168s 90ms/step - loss: 0.3369 - acc: 0.8938 - val_loss: 0.0910 - val_acc: 0.9712
    Epoch 2/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.1263 - acc: 0.9617Epoch 1/20
    1875/1875 [==============================] - 165s 88ms/step - loss: 0.1263 - acc: 0.9617 - val_loss: 0.0523 - val_acc: 0.9819
    Epoch 3/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.1028 - acc: 0.9691Epoch 1/20
    1875/1875 [==============================] - 166s 88ms/step - loss: 0.1028 - acc: 0.9691 - val_loss: 0.0481 - val_acc: 0.9837
    Epoch 4/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0890 - acc: 0.9740Epoch 1/20
    1875/1875 [==============================] - 167s 89ms/step - loss: 0.0889 - acc: 0.9740 - val_loss: 0.0395 - val_acc: 0.9872
    Epoch 5/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0833 - acc: 0.9754Epoch 1/20
    1875/1875 [==============================] - 165s 88ms/step - loss: 0.0834 - acc: 0.9753 - val_loss: 0.0405 - val_acc: 0.98668s
    Epoch 6/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0783 - acc: 0.9770Epoch 1/20
    1875/1875 [==============================] - 166s 88ms/step - loss: 0.0783 - acc: 0.9770 - val_loss: 0.0363 - val_acc: 0.9888
    Epoch 7/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0734 - acc: 0.9783Epoch 1/20
    1875/1875 [==============================] - 164s 88ms/step - loss: 0.0734 - acc: 0.9783 - val_loss: 0.0373 - val_acc: 0.9875
    Epoch 8/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0716 - acc: 0.9791Epoch 1/20
    1875/1875 [==============================] - 169s 90ms/step - loss: 0.0715 - acc: 0.9791 - val_loss: 0.0280 - val_acc: 0.9904
    Epoch 9/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0659 - acc: 0.9805Epoch 1/20
    1875/1875 [==============================] - 167s 89ms/step - loss: 0.0659 - acc: 0.9805 - val_loss: 0.0285 - val_acc: 0.9909
    Epoch 10/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0672 - acc: 0.9803Epoch 1/20
    1875/1875 [==============================] - 166s 89ms/step - loss: 0.0672 - acc: 0.9803 - val_loss: 0.0266 - val_acc: 0.9923
    Epoch 11/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0652 - acc: 0.9811Epoch 1/20
    1875/1875 [==============================] - 165s 88ms/step - loss: 0.0652 - acc: 0.9811 - val_loss: 0.0325 - val_acc: 0.9889
    Epoch 12/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0631 - acc: 0.9819- ETA: 0s - loss: 0.0632 - acEpoch 1/20
    1875/1875 [==============================] - 166s 89ms/step - loss: 0.0630 - acc: 0.9819 - val_loss: 0.0286 - val_acc: 0.9906
    Epoch 13/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0632 - acc: 0.9814Epoch 1/20
    1875/1875 [==============================] - 167s 89ms/step - loss: 0.0631 - acc: 0.9814 - val_loss: 0.0281 - val_acc: 0.9912
    Epoch 14/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0620 - acc: 0.9812- ETA: 0s - loss: 0.0620 - acc:Epoch 1/20
    1875/1875 [==============================] - 164s 88ms/step - loss: 0.0619 - acc: 0.9812 - val_loss: 0.0257 - val_acc: 0.9917
    Epoch 15/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0601 - acc: 0.9821- ETA: 1s - loss: 0.0602 - acc: 0 - ETA: 1s - loss: 0.0602 - acc: 0.98 - ETA: 1s - loss: 0.06Epoch 1/20
    1875/1875 [==============================] - 166s 88ms/step - loss: 0.0601 - acc: 0.9821 - val_loss: 0.0245 - val_acc: 0.9911
    Epoch 16/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0603 - acc: 0.9828Epoch 1/20
    1875/1875 [==============================] - 166s 89ms/step - loss: 0.0603 - acc: 0.9828 - val_loss: 0.0237 - val_acc: 0.9924
    Epoch 17/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0608 - acc: 0.9822Epoch 1/20
    1875/1875 [==============================] - 166s 89ms/step - loss: 0.0608 - acc: 0.9822 - val_loss: 0.0237 - val_acc: 0.9920
    Epoch 18/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0570 - acc: 0.9839Epoch 1/20
    1875/1875 [==============================] - 166s 88ms/step - loss: 0.0572 - acc: 0.9839 - val_loss: 0.0239 - val_acc: 0.9930
    Epoch 19/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0559 - acc: 0.9840Epoch 1/20
    1875/1875 [==============================] - 167s 89ms/step - loss: 0.0558 - acc: 0.9840 - val_loss: 0.0259 - val_acc: 0.9923
    Epoch 20/20
    1874/1875 [============================>.] - ETA: 0s - loss: 0.0595 - acc: 0.9827Epoch 1/20
    1875/1875 [==============================] - 168s 90ms/step - loss: 0.0595 - acc: 0.9826 - val_loss: 0.0323 - val_acc: 0.9913
    

<a id="Accuracy-Comparison"></a>
### 4.4.3. Accuracy Comparison


```python
x=cnn_augment_run.epoch
fig, ax = plt.subplots(figsize=fsize)
ax = sns.lineplot(x, cnn_run.history['acc'], color='b', label='Training accuracy - no data augm')
ax = sns.lineplot(x, cnn_run.history['val_acc'], color='g', label='Validation accuracy - no data augm')
ax = sns.lineplot(x, cnn_augment_run.history['acc'], color='gray', label='Training accuracy - w/ data augment')
ax = sns.lineplot(x, cnn_augment_run.history['val_acc'], color='black', label='Validation accuracy - w/ data augment')
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_71_0.png" style="max-width:840px;">
</p>



```python
train_acc=cnn_augment.evaluate(X_train, Y_train)[1]
test_acc=cnn_augment.evaluate(X_test, Y_test)[1]
print('\n\nTrain accuracy: {}% over {} epochs.'.format(round(train_acc*100,2),len(nn_nodrop_run.history)))
print('Test accuracy: {}% over {} epochs.'.format(round(test_acc*100,2),len(nn_nodrop_run.history)))
df_res=df_res.append({'Classifier': 'CNN - w/ data augment','Training accuracy':train_acc,'Testing accuracy':test_acc}, ignore_index=True)
```

    60000/60000 [==============================] - 55s 921us/sample - loss: 0.0159 - acc: 0.9952
    10000/10000 [==============================] - 9s 932us/sample - loss: 0.0185 - acc: 0.9946
    
    
    Train accuracy: 99.52% over 4 epochs.
    Test accuracy: 99.46% over 4 epochs.
    

<a id="Error-Analysis"></a>
## 4.5. Error Analysis

The neural network with dropout resulted in the best results. Let's display examples that were not properly categorized.


```python
X_test_pred=cnn_augment.predict(X_test).argmax(axis=1)
y_pred=X_test_pred[X_test_pred!=Y_test]
y_actual=Y_test[X_test_pred!=Y_test]
x=X_test0[X_test_pred!=Y_test]
```


```python
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
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_76_0.png" style="max-width:840px;">
</p>



```python
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
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_77_0.png" style="max-width:840px;">
</p>



```python

```


```python

```


```python
# Plot confusion matrix
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
y_pred=X_test_pred[X_test_pred!=Y_test]
y_actual=Y_test[y_test_pred!=Y_test]
x=X_test0[X_test_pred!=Y_test]

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
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_80_0.png" style="max-width:840px;">
</p>



```python
# The following code is taken from https://github.com/slundberg/shap/blob/b606ab179d5b70ec6bd3e5acbaaed4c9bd65a14e/shap/plots/image.py
# and slightly edited for better appearance of the colorbar. All credits to original author. 


```


```python
try:
    nbackgroundsamples=min(n,1000) # <--- run time is proportionnal to that number!
except:
    nbackgroundsamples=1000
ntodisplay = 10
np.random.seed(1986)

# select a set of background examples to take an expectation over
rand=np.random.choice(X_train.shape[0], nbackgroundsamples, replace=False)
background = X_train[rand]
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

shap_values = e.shap_values(X_select)

# plot the feature attributions
        
SD_tools.plot_shap_values(shap_values, -X_select)
```


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_82_0.png" style="max-width:840px;">
</p>


__Analysis:__

In the above plots, ten examples correctly predicted, one for each class, are shown (see the bold number at the beginning of each line). Each column shows what the model "sees" when it's trying to predict if it belongs to each of the class. Each column represent a potential class, from 0 on the left to 9 on the right. 

Red pixels indicates that what the models sees it favorable for the class considered, blue means it's unfavorable. 

Examples:
- For the 0, the center of the image where there is nothing is important.
- For the 4: it is very important that there is no top bar at the top of the 4, otherwise it would be a 9. You can actually see that the top bar is blue in the 9 column.
- Same for the 6, it is important that the top area of the digit be clear: it is blue in the 0 column and red in the 6 column.


```python
import shap
import numpy as np

try:
    nbackgroundsamples=min(n,1000) # <--- run time is proportionnal to that number!
except:
    nbackgroundsamples=1000
    
ntodisplay = 10
np.random.seed(1986)

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
print(Y_select)

shap_values = e.shap_values(X_select)

# plot the feature attributions
        
SD_tools.plot_shap_values(shap_values, -X_select)
```

    [9, 7, 1, 8, 6, 3, 0, 9, 3, 4]
    


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_84_1.png" style="max-width:840px;">
</p>


This is plot is the same as the previous one except it shows miscategorized digits. There are several distincts problems that are appearing:
- digits where some of the "ink" is worn out. For instance the 0 and the 8. Human being are good at reading these numbers because we can easily predict for the existing lines fading which lines are missing,
- digits where the typical proportions are not respected, like the 5, 6 and 7 shown above.

A potential solution could be to increase the complexity of the convolutional neural network and to extend the data augmentation. For instance for the 6 shown above, if it was rotated 30° to the right it would have been recognized.

An other possibility would be to create or obtain more data but this would be typically expensive in terms of efforts and should not be the main priority. 


```python
# Visualizating filters
# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

SD_tools.visualize_filters(model = cnn_augment, img = np.array(X_train[10]).reshape((1, 28, 28, 1)).astype(np.float64), 
                      layer_name = 'conv2d_2', print_summary=1, h_pad=0.05)
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_2 (Conv2D)            (None, 28, 28, 32)        1600      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 14, 14, 48)        38448     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 48)          0         
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 7, 7, 48)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 2352)              0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 128)               301184    
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 342,522
    Trainable params: 342,522
    Non-trainable params: 0
    _________________________________________________________________
    
    


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_86_1.png" style="max-width:840px;">
</p>



```python
SD_tools.visualize_filters(model = cnn_augment, img = np.array(X_train[10]).reshape((1, 28, 28, 1)).astype(np.float64), 
                      layer_name = 'max_pooling2d_2', print_summary=1, h_pad=0.05)
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_2 (Conv2D)            (None, 28, 28, 32)        1600      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 14, 14, 48)        38448     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 48)          0         
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 7, 7, 48)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 2352)              0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 128)               301184    
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 342,522
    Trainable params: 342,522
    Non-trainable params: 0
    _________________________________________________________________
    
    


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_87_1.png" style="max-width:840px;">
</p>



```python
SD_tools.visualize_filters(model = cnn_augment, img = np.array(X_train[10]).reshape((1, 28, 28, 1)).astype(np.float64), 
                      layer_name = 'conv2d_3', print_summary=1, h_pad=0.05)
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_2 (Conv2D)            (None, 28, 28, 32)        1600      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 14, 14, 48)        38448     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 48)          0         
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 7, 7, 48)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 2352)              0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 128)               301184    
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 342,522
    Trainable params: 342,522
    Non-trainable params: 0
    _________________________________________________________________
    
    


<p align="center">
    <img src="https://sdamolini.github.io/assets/img/MNIST/output_88_1.png" style="max-width:840px;">
</p>


<a id="Summary"></a>
# 5. Summary


```python
print('The training and test accuracies for all models in thus notebook are shwon below:')
df_res.round(5)
```

    The training and test accuracies for all models in thus notebook are shwon below:
    




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
      <td>0.91437</td>
      <td>0.9046</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear - Ridge</td>
      <td>0.92287</td>
      <td>0.9175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Linear - ElasticNet</td>
      <td>0.92023</td>
      <td>0.9147</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NN - no dopout</td>
      <td>0.99477</td>
      <td>0.9796</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NN - w/ dropout</td>
      <td>0.99450</td>
      <td>0.9812</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CNN - no data augment</td>
      <td>0.99903</td>
      <td>0.9941</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CNN - w/ data augment</td>
      <td>0.99523</td>
      <td>0.9946</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('The model with the highest testing accuracy is {} with {.2f}%'.format())
```


```python

```
