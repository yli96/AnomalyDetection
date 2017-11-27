#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:08:32 2017

@author: yueningli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from tempfile import TemporaryFile
oneMiniteWindow=np.zeros(24*60)
oneMiniteDuplWindow=np.zeros(24*60*2)
fiveMinutesWindow=np.zeros(24*60/5)
fiveMinutesSlidingWindow=np.zeros(24*60-4)
tenMinutesWindow=np.zeros(24*60/10)
tenMinutesSlidingWindow=np.zeros(24*60-9)
data = pd.read_csv('shanghai.csv')
shanghai = TemporaryFile()
length=len(data)
logtime=data['_time']
k=pd.Series(0)
q=np.zeros(2400)
for i in range(0,length): 
    k[i]=int(str(logtime[i][11:13]+''+logtime[i][14:16]))
    a=k[i]
    q[a]=q[a]+1

minutereg=np.zeros(length)
for i in range(0,length):
    minutereg[i]=k[i]-((k[i]/100)*40)
x_d=np.linspace(0, 1440, 1440)
col=16
c=0

for i in range(0,len(q)):
    if i%100<60:
        oneMiniteWindow[c]=q[i]
        c=c+1
for i in range(0,24*60/5):
    fiveMinutesWindow[i]=oneMiniteWindow[i*5]+oneMiniteWindow[i*5+1]+oneMiniteWindow[i*5+2]+oneMiniteWindow[i*5+3]+oneMiniteWindow[i*5+4] 
for i in range(0,24*60-4):
    fiveMinutesSlidingWindow[i]=oneMiniteWindow[i]+oneMiniteWindow[i+1]+oneMiniteWindow[i+2]+oneMiniteWindow[i+3]+oneMiniteWindow[i+4]
for i in range(0,24*60/10):
    tenMinutesWindow[i]=fiveMinutesWindow[i*2]+fiveMinutesWindow[i*2+1]
for i in range(0,24*60-9):
    tenMinutesSlidingWindow[i]=oneMiniteWindow[i]+oneMiniteWindow[i+1]+oneMiniteWindow[i+2]+oneMiniteWindow[i+3]+oneMiniteWindow[i+4]+oneMiniteWindow[i+5]+oneMiniteWindow[i+6]+oneMiniteWindow[i+7]+oneMiniteWindow[i+8]+oneMiniteWindow[i+9]

#oneMiniteDuplWindow[0:len(oneMiniteWindow)]=oneMiniteWindow
#oneMiniteDuplWindow[len(oneMiniteWindow):2*len(oneMiniteWindow)]=oneMiniteWindow
kde = KernelDensity(bandwidth=40.0, kernel='gaussian')
kde.fit(minutereg[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

#plt.fill_between(x_d, np.exp(logprob), alpha=0.5)

plt.fill_between(x_d, np.exp(logprob)/max(np.exp(logprob)), alpha=0.5)
plt.plot(minutereg, np.full_like(minutereg, -0.01), '|k', markeredgewidth=1)
plt.ylim(0, 1)
np.save('shanghai.npy',np.exp(logprob)/max(np.exp(logprob)))
np.load('shanghai.npy')
