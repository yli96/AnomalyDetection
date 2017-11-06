#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:33:04 2017

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
from sklearn.ensemble import IsolationForest
data = pd.read_csv('individualAnomaly2.csv')
length=len(data)
col=16
logtime=data['_time']
s=pd.Series(length)
j=pd.Series(length)
k=pd.Series(length)
for i in range(0,length): 
    s[i]=logtime[i][11:19]
    j[i]=logtime[i][0:10]
    k[i]=int(str(logtime[i][11:13]+''+logtime[i][14:16]+''+logtime[i][17:19]))
timing = pd.DataFrame({"date":j,'time':s})
timing['datetime']=pd.to_datetime(timing['date'].astype(str)+' '+timing['time'].astype(str))

X = np.reshape(k,(-1,1))
clf = IsolationForest(max_samples=length, random_state=rng)
clf.fit(X)
y_pred = clf.predict(X)
b=clf.decision_function(X)
for i in range(0,length):
    if b[i]<-0.1:
        print 'Anomaly Detected at:', i

a=np.arange(length)
plt.figure(1)
plt.plot(a,b)
