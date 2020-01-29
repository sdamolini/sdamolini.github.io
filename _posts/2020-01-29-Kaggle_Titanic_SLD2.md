---
layout: post
title: "Kaggle - Titanic Competition"
img: TITANIC_background.jpg
date: 2017-07-04 12:54:00 +0300
description: None. 
tag: [Travel, Texas, Canyon]
---
# Introduction

This notebook is a take on the legendary Kaggle Titanic Machine Learning competition. 

RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in 1912 after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. (Wikipedia)


In this Kaggle challenge, the goal is to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

<img src="https://live.staticflickr.com/3397/3279461836_078feb313b_b.jpg">

# Imports

## Import Librairies and Tools

Let's import the packages needed to perform this analysis.


```python
import pandas as pd
print("Pandas version: {}".format(pd.__version__))

import numpy as np
print("Numpy version: {}".format(np.__version__))

import matplotlib
import matplotlib.pyplot as plt
print("Matplotlib version: {}".format(matplotlib.__version__))

import scipy as sp
print("Scipy version: {}".format(sp.__version__))

import sklearn
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
print("Sklearn version: {}".format(sklearn.__version__))

import xgboost
from xgboost import XGBClassifier
print("Xgboost version: {}".format(xgboost.__version__))

import seaborn as sns
print("Seaborn version: {}".format(sns.__version__))

import IPython
from IPython.display import display
print("IPython version: {}".format(IPython.__version__))

%matplotlib inline

# Do not show warnings (used when the notebook is completed)
import warnings
# warnings.filterwarnings('ignore') !!!
```

    Pandas version: 0.25.3
    Numpy version: 1.18.1
    Matplotlib version: 3.1.2
    Scipy version: 1.3.2
    Sklearn version: 0.22.1
    Xgboost version: 0.90
    Seaborn version: 0.9.0
    IPython version: 7.11.1
    