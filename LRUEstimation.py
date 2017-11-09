#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:25:34 2017

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

'''def readFile():'''
data = pd.read_csv('individualAnomaly2.csv')
length=len(data)
col=10
window=100
predict=np.zeros((length,col))
recent=np.zeros(window)
predictLRU=np.zeros((length-window+1,col))
#Newton Cooling
for i in range(0,window):
    recent[i]=1-np.exp(-0.02*i)
lru=np.zeros((length,col))
predict=np.zeros((length,col))
category=np.zeros((length,col))
sa=np.random.normal(mu,sigma,length) 
'''def loginStatusCheck(self):'''
for i in range(0,length):
    if data['STATUS'][i]!='SUCCESS':
        predict['STATUS'][i]=0
    else: predict[i][0]=1 
category[:,0]=1+sa
'''def operationSystemCheck(self):'''
osCount={}
for os in data['OPERATING_SYSTEM']:
    osCount[os]=osCount.get(os,0.0)+1.0
for i in range(0,length):
    if pd.isnull(data['OPERATING_SYSTEM'][i]):
        data.fillna(method='bfill')
    predict[i][1]=osCount[data['OPERATING_SYSTEM'][i]]/length
oslen=len(osCount)
ostmp=list(osCount)
for i in range(0,length):
    for j in range(0,oslen):
        if data['OPERATING_SYSTEM'][i]==ostmp[j]:
            sa=np.random.normal(mu,sigma)
            category[i][1]=j+1+sa
    
for start in range(0,length-window+1):  
    osCountWindow={}
    for os in data['OPERATING_SYSTEM'][start:start+window]:
        osCountWindow[os]=osCountWindow.get(os,0.0)+1.0
    osCountLRU=osCountWindow.copy()
    for i in range(0,window):
        osCountLRU[data['OPERATING_SYSTEM'][i]]=osCountLRU[data['OPERATING_SYSTEM'][i]]-recent[i]
    osLRUlen=len(osCountLRU)
    osLRUlist=list(osCountLRU)
    sum=0
    for i in range(0,osLRUlen):
        sum=osCountLRU[osLRUlist[i]]+sum
    if data['OPERATING_SYSTEM'][start+window-1] in osCountLRU.keys():
        predictLRU[start][1]=osCountLRU[data['OPERATING_SYSTEM'][start+window-1]]/sum
    else:
        predictLRU[start][1]=0
    '''
    for i in range(0,window):
        osCountLRU[data['OPERATING_SYSTEM'][i]]=osCountLRU[data['OPERATING_SYSTEM'][i]]/sum
    
    for i in range(0,len(predictLRU)):
        predictLRU[i][1]=osCountLRU[data['OPERATING_SYSTEM'][i]]/sum
    '''
'''def deviceTypeCheck(self):'''
deviceCount={}
for dt in data['DEVICE_TYPE']:
    deviceCount[dt]=deviceCount.get(dt,0.0)+1.0
for i in range(0,length):
    if pd.isnull(data['DEVICE_TYPE'][i]):
        data.fillna(method='bfill')
    predict[i][2]=deviceCount[data['DEVICE_TYPE'][i]]/length
dtlen=len(deviceCount)
dttmp=list(deviceCount)
for i in range(0,length):
    for j in range(0,dtlen):
        if data['DEVICE_TYPE'][i]==dttmp[j]:
            sa=np.random.normal(mu,sigma)
            category[i][2]=j+1+sa
for start in range(0,length-window+1):  
    dtCountWindow={}
    for dt in data['DEVICE_TYPE'][start:start+window]:
        dtCountWindow[dt]=dtCountWindow.get(dt,0.0)+1.0
    dtCountLRU=dtCountWindow.copy()
    for i in range(0,window):
        dtCountLRU[data['DEVICE_TYPE'][i]]=dtCountLRU[data['DEVICE_TYPE'][i]]-recent[i]
    dtLRUlen=len(dtCountLRU)
    dtLRUlist=list(dtCountLRU)
    sum=0
    for i in range(0,dtLRUlen):
        sum=dtCountLRU[dtLRUlist[i]]+sum
    if data['DEVICE_TYPE'][start+window-1] in dtCountLRU.keys():
        predictLRU[start][2]=dtCountLRU[data['DEVICE_TYPE'][start+window-1]]/sum
    else:
        predictLRU[start][2]=0
'''def browserCheck(self):'''
browserCount={}
for bc in data['BROWSER']:
    browserCount[bc]=browserCount.get(bc,0.0)+1.0
for i in range(0,length):
    if pd.isnull(data['BROWSER'][i]):
        data.fillna(method='bfill')
    predict[i][3]=browserCount[data['BROWSER'][i]]/length
bclen=len(browserCount)
bctmp=list(browserCount)
for i in range(0,length):
    for j in range(0,bclen):
        if data['BROWSER'][i]==bctmp[j]:
            sa=np.random.normal(mu,sigma)
            category[i][3]=j+1+sa
for start in range(0,length-window+1):  
    bcCountWindow={}
    for bc in data['BROWSER'][start:start+window]:
        bcCountWindow[bc]=bcCountWindow.get(bc,0.0)+1.0
    bcCountLRU=bcCountWindow.copy()
    for i in range(0,window):
        bcCountLRU[data['BROWSER'][i]]=bcCountLRU[data['BROWSER'][i]]-recent[i]
    bcLRUlen=len(bcCountLRU)
    bcLRUlist=list(bcCountLRU)
    sum=0
    for i in range(0,bcLRUlen):
        sum=bcCountLRU[bcLRUlist[i]]+sum
    if data['BROWSER'][start+window-1] in bcCountLRU.keys():
        predictLRU[start][3]=bcCountLRU[data['BROWSER'][start+window-1]]/sum
    else:
        predictLRU[start][3]=0
'''def connectionTypeCheck(self):'''
connectionType={}
for ct in data['client_connectionType']:
    connectionType[ct]=connectionType.get(ct,0.0)+1.0
for i in range(0,length):
    if pd.isnull(data['client_connectionType'][i]):
        data.fillna(method='bfill')
    predict[i][4]=connectionType[data['client_connectionType'][i]]/length
ctlen=len(connectionType)
cttmp=list(connectionType)
for i in range(0,length):
    for j in range(0,ctlen):
        if data['client_connectionType'][i]==cttmp[j]:
            sa=np.random.normal(mu,sigma)
            category[i][4]=j+1+sa
for start in range(0,length-window+1):  
    ctCountWindow={}
    for ct in data['client_connectionType'][start:start+window]:
        ctCountWindow[ct]=ctCountWindow.get(ct,0.0)+1.0
    ctCountLRU=ctCountWindow.copy()
    for i in range(0,window):
        ctCountLRU[data['client_connectionType'][i]]=ctCountLRU[data['client_connectionType'][i]]-recent[i]
    ctLRUlen=len(ctCountLRU)
    ctLRUlist=list(ctCountLRU)
    sum=0
    for i in range(0,ctLRUlen):
        sum=ctCountLRU[ctLRUlist[i]]+sum
    if data['client_connectionType'][start+window-1] in ctCountLRU.keys():
        predictLRU[start][4]=ctCountLRU[data['client_connectionType'][start+window-1]]/sum
    else:
        predictLRU[start][4]=0
'''def applicationNameCheck(self):'''
appCount={}
for an in data['APPLICATION_NAME']:
    appCount[an]=appCount.get(an,0.0)+1.0
for i in range(0,length):
    if pd.isnull(data['APPLICATION_NAME'][i]):
        data.fillna(method='bfill')
    predict[i][5]=appCount[data['APPLICATION_NAME'][i]]/length
aclen=len(appCount)
acnum=np.linspace(1,aclen,aclen)
actmp=list(appCount)
for i in range(0,length):
    for j in range(0,aclen):
        if data['APPLICATION_NAME'][i]==actmp[j]:
            sa=np.random.normal(mu,sigma)
            category[i][5]=j+1+sa
for start in range(0,length-window+1):  
    acCountWindow={}
    for ac in data['APPLICATION_NAME'][start:start+window]:
        acCountWindow[ac]=acCountWindow.get(ac,0.0)+1.0
    acCountLRU=acCountWindow.copy()
    for i in range(0,window):
        acCountLRU[data['APPLICATION_NAME'][i]]=acCountLRU[data['APPLICATION_NAME'][i]]-recent[i]
    acLRUlen=len(acCountLRU)
    acLRUlist=list(acCountLRU)
    sum=0
    for i in range(0,acLRUlen):
        sum=acCountLRU[acLRUlist[i]]+sum
    if data['APPLICATION_NAME'][start+window-1] in acCountLRU.keys():
        predictLRU[start][5]=acCountLRU[data['APPLICATION_NAME'][start+window-1]]/sum
    else:
        predictLRU[start][5]=0
'''def clientCountryCheck(self):'''
clientCountry={}
for ccoun in data['client_country']:
    clientCountry[ccoun]=clientCountry.get(ccoun,0.0)+1.0
for i in range(0,length):
    if pd.isnull(data['client_country'][i]):
        data.fillna(method='bfill')
    predict[i][6]=clientCountry[data['client_country'][i]]/length
ccounlen=len(clientCountry)
ccountmp=list(clientCountry)
for i in range(0,length):
    for j in range(0,ccounlen):
        if data['client_country'][i]==ccountmp[j]:
            sa=np.random.normal(mu,sigma)
            category[i][6]=j+1+sa
for start in range(0,length-window+1):  
    ccounCountWindow={}
    for ccoun in data['client_country'][start:start+window]:
        ccounCountWindow[ccoun]=ccounCountWindow.get(ccoun,0.0)+1.0
    ccounCountLRU=ccounCountWindow.copy()
    for i in range(0,window):
        ccounCountLRU[data['client_country'][i]]=ccounCountLRU[data['client_country'][i]]-recent[i]
    ccounLRUlen=len(ccounCountLRU)
    ccounLRUlist=list(ccounCountLRU)
    sum=0
    for i in range(0,ccounLRUlen):
        sum=ccounCountLRU[ccounLRUlist[i]]+sum
    if data['client_country'][start+window-1] in ccounCountLRU.keys():
        predictLRU[start][6]=ccounCountLRU[data['client_country'][start+window-1]]/sum
    else:
        predictLRU[start][6]=0
'''def clientCityCheck(self):'''
clientCity={}
for ccity in data['client_city']:
    clientCity[ccity]=clientCity.get(ccity,0.0)+1.0
for i in range(0,length):
    if pd.isnull(data['client_city'][i]):
        data.fillna(method='bfill')
    predict[i][7]=clientCity[data['client_city'][i]]/length
ccitylen=len(clientCity)
ccitytmp=list(clientCity)
for i in range(0,length):
    for j in range(0,ccitylen):
        if data['client_city'][i]==ccitytmp[j]:
            sa=np.random.normal(mu,sigma)
            category[i][7]=j+1+sa
for start in range(0,length-window+1):  
    ccityCountWindow={}
    for ccity in data['client_city'][start:start+window]:
        ccityCountWindow[ccity]=ccityCountWindow.get(ccity,0.0)+1.0
    ccityCountLRU=ccityCountWindow.copy()
    for i in range(0,window):
        ccityCountLRU[data['client_city'][i]]=ccityCountLRU[data['client_city'][i]]-recent[i]
    ccityLRUlen=len(ccityCountLRU)
    ccityLRUlist=list(ccityCountLRU)
    sum=0
    for i in range(0,ccityLRUlen):
        sum=ccityCountLRU[ccityLRUlist[i]]+sum
    if data['client_city'][start+window-1] in ccityCountLRU.keys():
        predictLRU[start][7]=ccityCountLRU[data['client_city'][start+window-1]]/sum
    else:
        predictLRU[start][7]=0
'''def clientISPCheck(self):'''
clientISP={}
for cisp in data['client_isp']:
    clientISP[cisp]=clientISP.get(cisp,0.0)+1.0
for i in range(0,length):
    if pd.isnull(data['client_isp'][i]):
        data.fillna(method='bfill')
    predict[i][8]=clientISP[data['client_isp'][i]]/length
cisplen=len(clientISP)
cisptmp=list(clientISP)
for i in range(0,length):
    for j in range(0,cisplen):
        if data['client_isp'][i]==cisptmp[j]:
            sa=np.random.normal(mu,sigma)
            category[i][8]=j+1+sa
for start in range(0,length-window+1):  
    cispCountWindow={}
    for cisp in data['client_isp'][start:start+window]:
        cispCountWindow[cisp]=cispCountWindow.get(cisp,0.0)+1.0
    cispCountLRU=cispCountWindow.copy()
    for i in range(0,window):
        cispCountLRU[data['client_isp'][i]]=cispCountLRU[data['client_isp'][i]]-recent[i]
    cispLRUlen=len(cispCountLRU)
    cispLRUlist=list(cispCountLRU)
    sum=0
    for i in range(0,cispLRUlen):
        sum=cispCountLRU[cispLRUlist[i]]+sum
    if data['client_isp'][start+window-1] in cispCountLRU.keys():
        predictLRU[start][8]=cispCountLRU[data['client_isp'][start+window-1]]/sum
    else:
        predictLRU[start][8]=0
'''def clientDomainCheck(self):
    clientDomain={}
    for cd in data['client_domain']:
        clientDomain[cd]=clientDomain.get(cd,0.0)+1.0
    for i in range(0,length-1):
        predict[i][9]=clientDomain[data['client_domain'][i]]/length'''
'''def clientOrganizationCheck(self):'''
clientOrganization={}
for co in data['client_organization']:
    clientOrganization[co]=clientOrganization.get(co,0.0)+1.0
for i in range(0,length):
    if pd.isnull(data['client_organization'][i]):
        data.fillna(method='bfill')
    predict[i][9]=clientOrganization[data['client_organization'][i]]/length 
colen=len(clientOrganization)
cotmp=list(clientOrganization)
for i in range(0,length):
    for j in range(0,colen):
        if data['client_organization'][i]==cotmp[j]:
            sa=np.random.normal(mu,sigma)
            category[i][9]=j+1+sa 
for start in range(0,length-window+1):  
    coCountWindow={}
    for co in data['client_organization'][start:start+window]:
        coCountWindow[co]=coCountWindow.get(co,0.0)+1.0
    coCountLRU=coCountWindow.copy()
    for i in range(0,window):
        coCountLRU[data['client_organization'][i]]=coCountLRU[data['client_organization'][i]]-recent[i]
    coLRUlen=len(coCountLRU)
    coLRUlist=list(coCountLRU)
    sum=0
    for i in range(0,coLRUlen):
        sum=coCountLRU[coLRUlist[i]]+sum
    if data['client_organization'][start+window-1] in coCountLRU.keys():
        predictLRU[start][9]=coCountLRU[data['client_organization'][start+window-1]]/sum
    else:
        predictLRU[start][9]=0
