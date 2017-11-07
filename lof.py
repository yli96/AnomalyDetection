#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:17:48 2017

@author: yueningli
"""

import sys
import os
import csv
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
data = pd.read_csv('individualAnomaly.csv')
length=len(data)
col=16


'''np.random.seed(42)'''

# Generate train data
X = np.vstack((data['client_lat'],data['client_lon']))
X = X.T
'''# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X + 2, X - 2, X_outliers]'''

# fit the model
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)
for i in range(0,length):
    if y_pred[i]<0:
        print 'Anomaly Detected at:', i

