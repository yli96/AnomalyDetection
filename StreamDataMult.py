#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:55:27 2017

@author: yueningli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

os.chdir('/Users/yueningli/Documents/NetworkAnomaly/ParzenWindow')
data=pd.read_csv('login.csv')
clientCountry={}
for ccoun in data['client_country']:
    clientCountry[ccoun]=clientCountry.get(ccoun,0.0)+1.0
for co in clientCountry:
    if pd.isnull(co)==0:
        country=data.loc[data['client_country']==co]
        oneMiniteWindow=np.zeros(24)
        length=len(country)
        logtime=country['_time']
        k=pd.Series(0)
        q=np.zeros(24)
        for i in range(0,length): 
            k[i]=int(str(logtime[logtime.index[i]][11:13]))
            q[k[i]]+=1
        x_d=np.linspace(0, 23, 24)
        col=16
        c=0
        #oneMiniteDuplWindow[0:len(oneMiniteWindow)]=oneMiniteWindow
        #oneMiniteDuplWindow[len(oneMiniteWindow):2*len(oneMiniteWindow)]=oneMiniteWindow
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian').fit(k[:, None])
        # score_samples returns the log of the probability density
        logprob = kde.score_samples(x_d[:, None])
        
        #plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
        
        title=co+'.npy'
        np.save(title,np.exp(logprob)/max(np.exp(logprob)))

'''
os.chdir('/Users/yueningli/Documents/NetworkAnomaly/ParzenWindow')
data=pd.read_csv('login.csv')
clientCountry={}
for ccoun in data['client_country']:
    clientCountry[ccoun]=clientCountry.get(ccoun,0.0)+1.0
for co in clientCountry:
    if pd.isnull(co)==0:
        country=data.loc[data['client_country']==co]
        oneMiniteWindow=np.zeros(24)
        length=len(country)
        logtime=country['_time']
        k=pd.Series(0)
        q=np.zeros(24)
        for i in range(0,length): 
            k[i]=int(str(logtime[i][11:13]))
            q[k[i]]+=1
        x_d=np.linspace(0, 23, 24)
        col=16
        c=0
        #oneMiniteDuplWindow[0:len(oneMiniteWindow)]=oneMiniteWindow
        #oneMiniteDuplWindow[len(oneMiniteWindow):2*len(oneMiniteWindow)]=oneMiniteWindow
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian').fit(k[:, None])
        # score_samples returns the log of the probability density
        logprob = kde.score_samples(x_d[:, None])
        
        #plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
        
        plt.fill_between(x_d, np.exp(logprob)/max(np.exp(logprob)), alpha=0.5)
        plt.plot(minutereg, np.full_like(minutereg, -0.01), '|k', markeredgewidth=1)
        plt.ylim(0, 1)
        title=co+'.npy'
        np.save(title,np.exp(logprob)/max(np.exp(logprob)))
'''
