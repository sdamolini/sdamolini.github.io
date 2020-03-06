---
layout: post
title: "Tutorial - Install TensorFlow GPU on Windows 10"
img:     TF_thumbnail.jpg
date: 2020-03-06 12:54:00 +0300
description: None. 
tag: ['Tensorflow', 'GPU', 'NVIDIA', 'cuDNN', 'CUDA']
---
<a id="Introduction"></a>
# 1. Introduction

Running Tensorflow using your computer GPU can save you a lot of time when performing complex calculations involving matrices. You may wonder, why are GPUs so much better than CPU for deep learning? 

Let's start with the CPU. It stands for Central Processing Unit and it has relatively few processor cores, typically 4 or 8 these days. That means the CPU is excellent are running complex calculations but it can only do it a few at a time.

Now the GPU: it stands for Graphics Processing Unit and it is is good at doing a fairly simple calculation on large numbers of variables in parallell. For instance, if you have a large matrix and you want to to an element-wise multiplication with a matrix of the same size, that's easy for a GPU. This comes from the fact that GPU are made to compute pixels on a screen, which requires a lot of easy calculations.

Deep learning, in particular Neural Networks (NN) and Convoluted Neural Networks (CNN) involve an enormous amount of relatively simple matrix operations and therefore the GPUs are excellent at this task.

<a id="Prerequisites"></a>
# 2. Prerequisites

<a id="Identify-Graphic-Card"></a>
## 2.1. Identify Graphic Card

This tutorial is for a Windows 10 machine with an Nvidia graphic card. The first thing you need to do is verify that your graphic card is able to run Tensorflow.

To see which graphic card you have:
- hit the WIndow key on your keyboard and search for System Information
- expand "Components" and click on "Display"

On the right panel, look at the name of the graphic cards. On most laptops, there is an integrated graphic chip, usually intel, and a dedicated grpahic card, in my case, an NVIDIA Geforce GTX 660M.

Once you've identified your graphic card, go to https://developer.nvidia.com/cuda-gpus and see if your graphic card made it. You can also look at the "Compute Capability" which will indicate how good your card will be. Mine is a 3.0 which isn't great but I'm happy my card made it on the list so I won't complain!


<a id="Update-your-graphic-card-drivers"></a>
## 2.2. Update your graphic card drivers

Now that you know the name of your graphic card, go to https://www.nvidia.com/Download/index.aspx and download and install the latest drivers for your card.

Reboot after installing.

<a id="Settle-on-a-Tensorflow-version"></a>
## 2.3. Settle on a Tensorflow version

Knowing which tensorflow version you want to use is very important to be sure all packages are compatible, but most importantly, ensure that Nvidia's CUDA and cuDNN are compatible.

To know which version of Tensorflow you currently use, type the following commands into a Jupyter Notebook.


```python
import tensorflow as tf
tf.__version__
```




    '1.14.0'



You may be happy the tensorflow version you are using or you may want to update to Tensoflow 2.0. That depends on what will your usage be. In my case, I realized a little too late that a package I really like (SHAP) stoppped functionning after upgrading to Tensorflow 2.0. I decided to settle with the latest Tensoflow 1 version, which is 1.14.
https://www.tensorflow.org/install/source_windows

<a id="Instal-Visual-Studio-Express"></a>
## 2.4. Instal Visual Studio Express

Visual is needed to run the NVIDIA frameworks. Install Visual Studio frome here: https://visualstudio.microsoft.com/vs/express/ and install it with the defaults options. Do not select any "workload" when offered, just continue without workloads.

Reboot your PC.

<a id="Get-the-required-CUDA-and-cuDNN-versions"></a>
## 2.5. Get the required CUDA and cuDNN versions

Once you know which Tensorflow version you want to use, head over to https://www.tensorflow.org/install/source_windows, and scroll down to the "GPU" table. Locate your tensorflow version and write down the Python version, CUDA version and cuDNN versions that are compatible.

Head over head to download the proper CUDA version: https://developer.nvidia.com/cuda-toolkit-archive

And go here to get the proper cuDNN version: https://developer.nvidia.com/rdp/cudnn-download
Note that you will need to create a FREE developer account to access that download page, it only takes a few seconds. 

<a id="Install-CUDA-and-cuDNN"></a>
## 2.6. Install CUDA and cuDNN


```python

```
