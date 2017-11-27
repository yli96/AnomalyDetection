#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:17:43 2017

@author: yueningli
"""



import sys
import os
#os.chdir('/Users/yueningli/Documents/NetworkAnomaly/ParzenWindow')
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
import time
data = pd.read_csv('login.csv')

idCount={}
for id in data['prsId']:
    idCount[id]=idCount.get(id,0.0)+1.0
#sort dictionary
idkey = sorted(idCount.iteritems(), key=lambda d:d[1], reverse = True)
#top n
n=300
topntemp=np.array(idkey[:n])
topnkey=topntemp[:,0]
#top n contains:
totalnum=sum(topntemp[:,1])-topntemp[0][1]

#user filter for Geolocation
print("Geolocation Detection")
user=pd.DataFrame
for num in range(1,n):
    user =data[data['prsId']==topnkey[num]]
    #print("Geolocation Detection--User:",num)
    length=len(user)
    col=16
    rng = np.random.RandomState(42)
    
    mu,sigma=0,0.0001
    sa=np.random.normal(mu,sigma,length)
    sb=np.random.normal(mu,sigma,length)
    # Generate train data
    X = np.vstack((user['client_lat'],user['client_lon']))
    
    X = X.T
    for i in range(0,length):
        if sp.isnan(X[i][0]):
            X[i][0]=200
    for i in range(0,length):
        if sp.isnan(X[i][1]):
            X[i][1]=200 
    X[:,0]=X[:,0]+sa
    X[:,1]=X[:,1]+sb
    start = time.clock()
    clf = IsolationForest(max_samples=length, random_state=rng)
    clf.fit(X)
    b=clf.decision_function(X)
    elapsed = (time.clock() - start)
    
    y_pred = clf.predict(X)
    
    for i in range(0,2*length):
        if b[i]<-0.2:
     #       print 'Anomaly Detected at:', user.index[i]
            print(user.index[i])
            print(user['prsId'][user.index[i]])
    '''
    i=0     
    while i < 2*length:
        if b[i]<-0.2:
            print(user.index[i])
            print(user['prsId'][user.index[i]])
            if i%2==0:
                i+=1
        i+=1
    '''
    #a=np.arange(length)
    #print("Time used:",elapsed)


#Wrong Timing
print("Wrong Timing Detection")
for num in range(1,n):
    user =data[data['prsId']==topnkey[num]]
    #print("Time Window Detection--User:",num)
    length=len(user)
    col=16
    logtime=user['_time']
    s=pd.Series(0)
    j=pd.Series(0)
    k=pd.Series(0)
    k=k.drop([0])
    for idx, val in enumerate(logtime): 
        #s[i]=logtime[i][11:19]
        #j[i]=logtime[i][0:10]
        k[logtime.index[idx]]=int(str(val[11:13]+''+val[14:16]))
    #timing = pd.DataFrame({"date":j,'time':s})
    #timing['datetime']=pd.to_datetime(timing['date'].astype(str)+' '+timing['time'].astype(str))
    kdual=pd.Series(0)
    kdual=kdual.drop([0])
    for i in range(0,length):
        kdual[k.index[i]]=k[k.index[i]]
        kdual[-k.index[i]]=k[k.index[i]]+240000
    mu,sigma=0,0.1
    sa=np.random.normal(mu,sigma,2*length)
    kdual=kdual+sa
    X = np.reshape(kdual,(-1,1))
    clf = IsolationForest(max_samples=2*length, random_state=rng)
    clf.fit(X)
    y_pred = clf.predict(X)
    b=clf.decision_function(X)
    
    for i in range(length/2,3*length/2):
        if b[i]<-0.25:
            #print 'Anomaly Detected at:', user.index[i]
            print abs(user.index[i/2])
            print(user['prsId'][user.index[i/2]])
    '''
    i=0     
    while i < 2*length:
        if b[i]<-0.25:
            print abs(user.index[i/2])
            print(user['prsId'][user.index[i/2]])
            if i%2==0:
                i+=1
        i+=1
    '''




#Categorical Data iForest

for num in range(1,n):
    user =data[data['prsId']==topnkey[num]]
    
    length=len(user)
    col=10
    mu,sigma=0,0.00001
    predict=np.zeros((length,col))   
    category=np.zeros((length,col))
    sa=np.random.normal(mu,sigma,length)
    
    '''def loginStatusCheck(self):'''
    for i in range(0,length):
        if user['STATUS'][user.index[i]]!='SUCCESS':
            predict['STATUS'][i]=0
        else: predict[i][0]=1 
    category[:,0]=1+sa
    '''def operationSystemCheck(self):'''
    osCount={}
    for os in user['OPERATING_SYSTEM']:
        osCount[os]=osCount.get(os,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['OPERATING_SYSTEM'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][1]=osCount[user['OPERATING_SYSTEM'][user.index[i]]]/length
    oslen=len(osCount)
    ostmp=list(osCount)
    for i in range(0,length):
        for j in range(0,oslen):
            if user['OPERATING_SYSTEM'][user.index[i]]==ostmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][1]=j+1+sa
    '''def deviceTypeCheck(self):'''
    deviceCount={}
    for dt in user['DEVICE_TYPE']:
        deviceCount[dt]=deviceCount.get(dt,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['DEVICE_TYPE'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][2]=deviceCount[user['DEVICE_TYPE'][user.index[i]]]/length
    dtlen=len(deviceCount)
    dttmp=list(deviceCount)
    for i in range(0,length):
        for j in range(0,dtlen):
            if user['DEVICE_TYPE'][user.index[i]]==dttmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][2]=j+1+sa
    '''def browserCheck(self):'''
    browserCount={}
    for bc in user['BROWSER']:
        browserCount[bc]=browserCount.get(bc,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['BROWSER'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][3]=browserCount[user['BROWSER'][user.index[i]]]/length
    bclen=len(browserCount)
    bctmp=list(browserCount)
    for i in range(0,length):
        for j in range(0,bclen):
            if user['BROWSER'][user.index[i]]==bctmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][3]=j+1+sa
    '''def connectionTypeCheck(self):'''
    connectionType={}
    for ct in user['client_connectionType']:
        connectionType[ct]=connectionType.get(ct,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_connectionType'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][4]=connectionType[user['client_connectionType'][user.index[i]]]/length
    ctlen=len(connectionType)
    cttmp=list(connectionType)
    for i in range(0,length):
        for j in range(0,ctlen):
            if user['client_connectionType'][user.index[i]]==cttmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][4]=j+1+sa
    '''def applicationNameCheck(self):'''
    appCount={}
    for an in user['APPLICATION_NAME']:
        appCount[an]=appCount.get(an,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['APPLICATION_NAME'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][5]=appCount[user['APPLICATION_NAME'][user.index[i]]]/length
    aclen=len(appCount)
    acnum=np.linspace(1,aclen,aclen)
    actmp=list(appCount)
    for i in range(0,length):
        for j in range(0,aclen):
            if user['APPLICATION_NAME'][user.index[i]]==actmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][5]=j+1+sa
    '''def clientCountryCheck(self):'''
    clientCountry={}
    for ccoun in user['client_country']:
        clientCountry[ccoun]=clientCountry.get(ccoun,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_country'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][6]=clientCountry[user['client_country'][user.index[i]]]/length
    ccounlen=len(clientCountry)
    ccountmp=list(clientCountry)
    for i in range(0,length):
        for j in range(0,ccounlen):
            if user['client_country'][user.index[i]]==ccountmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][6]=j+1+sa
    '''def clientCityCheck(self):'''
    clientCity={}
    for ccity in user['client_city']:
        clientCity[ccity]=clientCity.get(ccity,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_city'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][7]=clientCity[user['client_city'][user.index[i]]]/length
    ccitylen=len(clientCity)
    ccitytmp=list(clientCity)
    for i in range(0,length):
        for j in range(0,ccitylen):
            if user['client_city'][user.index[i]]==ccitytmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][7]=j+1+sa
    '''def clientISPCheck(self):'''
    clientISP={}
    for cisp in user['client_isp']:
        clientISP[cisp]=clientISP.get(cisp,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_isp'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][8]=clientISP[user['client_isp'][user.index[i]]]/length
    cisplen=len(clientISP)
    cisptmp=list(clientISP)
    for i in range(0,length):
        for j in range(0,cisplen):
            if user['client_isp'][user.index[i]]==cisptmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][8]=j+1+sa
    '''def clientDomainCheck(self):
        clientDomain={}
        for cd in user['client_domain']:
            clientDomain[cd]=clientDomain.get(cd,0.0)+1.0
        for i in range(0,length-1):
            predict[i][9]=clientDomain[user['client_domain'][i]]/length'''
    '''def clientOrganizationCheck(self):'''
    clientOrganization={}
    for co in user['client_organization']:
        clientOrganization[co]=clientOrganization.get(co,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_organization'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][9]=clientOrganization[user['client_organization'][user.index[i]]]/length 
    colen=len(clientOrganization)
    cotmp=list(clientOrganization)
    for i in range(0,length):
        for j in range(0,colen):
            if user['client_organization'][user.index[i]]==cotmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][9]=j+1+sa 
    start = time.clock()
    clf = IsolationForest(max_samples=length, random_state=rng)
    clf.fit(category)
    y_pred = clf.predict(category)
    
    b=clf.decision_function(category)
    elapsed = (time.clock() - start)
    
    y_pred = clf.predict(category)
    
    for i in range(0,length):
        if b[i]<-0.2:
            #print 'Anomaly Detected at:', i
            print user.index[i]
            print user['prsId'][user.index[i]]
        

