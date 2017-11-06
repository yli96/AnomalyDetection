#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:04:27 2017

@author: yueningli
"""

import sys
import os
import csv
import math
import random
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv('individualAnomaly2.csv')
length=len(data)
col=16
rng = np.random.RandomState(42)

'''np.random.seed(42)'''

# Generate train data
X = np.vstack((data['client_lat'],data['client_lon']))
X = X.T
for i in range(0,length):
    if sp.isnan(X[i][0]):
        X[i][0]=200
for i in range(0,length):
    if sp.isnan(X[i][1]):
        X[i][1]=200 
clf = IsolationForest(max_samples=length, random_state=rng)
clf.fit(X)
y_pred = clf.predict(X)
b=clf.decision_function(X)


for i in range(0,length):
    if b[i]<-0.1:
        print 'Anomaly Detected at:', i
a=np.arange(length)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =X[:,0]
y =X[:,1]
z =a



ax.scatter(x, y, z)


ax.set_xlabel('client_lat')
ax.set_ylabel('client_lon')
ax.set_zlabel('Row')
plt.show()

'''
x=np.arange(length)
plt.figure(1)
plt.plot(x,X[0],X[1])'''

'''
for i in range(0,length):
    if y_pred[i]<0:
        print 'Anomaly Detected at:', i
'''