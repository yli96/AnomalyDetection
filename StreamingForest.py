#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:17:44 2017

@author: yueningli
"""
from anytree import AnyNode, RenderTree

import sys
import os
import csv
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
mu,sigma=0,0.00001
os.chdir('/Users/yueningli/Documents/NetworkAnomaly/Anomaly')
'''def readFile():'''
data = pd.read_csv('individualAnomaly2.csv')
length=len(data)


'''
class Node:
    def __init__(self, val):
        self.Left=None
        self.Right=None
        self.value=0
        self.k=0

class Tree:
    def __init__(self):
        self.root = None

    def getRoot(self):
        return self.root

    def add(self, val):
        if(self.root == None):
            self.root = Node(val)
        else:
            self._add(val, self.root)

    def _add(self, val, node):
        if(val < node.v):
            if(node.l != None):
                self._add(val, node.l)
            else:
                node.l = Node(val)
        else:
            if(node.r != None):
                self._add(val, node.r)
            else:
                node.r = Node(val)

    def find(self, val):
        if(self.root != None):
            return self._find(val, self.root)
        else:
            return None

    def _find(self, val, node):
        if(val == node.v):
            return node
        elif(val < node.v and node.l != None):
            self._find(val, node.l)
        elif(val > node.v and node.r != None):
            self._find(val, node.r)

    def deleteTree(self):
        # garbage collector will do this for us. 
        self.root = None

    def printTree(self):
        if(self.root != None):
            self._printTree(self.root)

    def _printTree(self, node):
        if(node != None):
            self._printTree(node.l)
            print str(node.v) + ' '
            self._printTree(node.r)

def BuildSingleHSTree(k):
    if k == maxDepth:
        return Node(r=0,l=0)
    else:
        q = random.randrange(0,col+1, 1)
        p = (maxq+minq)/2
        temp = maxq
        maxq = p
        Left = BuildSingleHSTree(mmin,mmax,k+1)
        maxq = temp
        minq = p
        Right = BuildSingleHSTree(mmin,mmax,k+1)
        return Node(Left,Right,SplitAtt = q,SplitValue = p,r = 0,l = 0 )
'''
'''
result=''
col=9
for i in range(0, 4*col):
    q= random.randrange(0,col+1,1)
    result=str(result)+str(q)


col=12
class Node:
    def __init__(self, val=-1,Left=None,rchild=None):
        self.Left=Left
        self.Right=Right
        self.value=value
        self.k=k

k=0
maxDepth=4*col
def CreateGrid(k):
    if k==maxDepth:
        return Node(Left=0,Right=0,k=k)
    else:
        q = random.randrange(0,col+1,1)
        Node.k=k
        Node.value=q
        Node.Left = CreateGrid(k+1)
        Node.Right = CreateGrid(k+1)

def BuildSingleHSTree(mmin,mmax,k):
    if k == maxDepth:
        return Node(r=0,l=0)
    else:
        q= random randrange(0,cate+1,1)
        p=(max(q)+
def UpdateMass(x, Node,referenceWindow):
    referenceWindow? Node.r++:Node.l++
    if (Node.k < maxDepth) then:
        Let Node' be the next level of Node that x tracerses'
        UpdateMass(x,Node',referenceWindow)
def StreamingHSTrees(w,t):
    Build HSTrees:Initializse Work Space and call Algorithm 1 for
    each tree
    Record the first refence maass profile in HS-Trees:
        for each tree T：:
            invoke UpdateMass(x,T.root,true) for each item x in the first
            x in the first w instances of the stream
    Count = 0
    while data stream continues do:
        Receive the next streaaming point x
        s = 0
        for each tree T in HSTrees do:
            s = s + Score(x, T){accumulate scores}
            UpdateMass(x, T.root, false){update mass l in T}
        end for
        Report s as the anomaly score for x
        Count+=1
        if Count == w then:
            Update model: Node.r = Node.l for every node with 
            non-zero mass r or l
            Reset Node.l = 0 for every node with non-zero mass l
            Count = 0
    end while
'''



'''
def readFile():
data = pd.read_csv('loginAnomaly.csv')
'''
length=len(data)
col=10
mu,sigma=0,0.00001
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
category=pd.DataFrame(category)

col=10
k=0
maxDepth=15


class Node:
    def __init__(self, value=None,Left=None,Right=None,k=0):
        self.Left=Left
        self.Right=Right
        self.value=value
        self.k=k


'''
def CreateGrid(k):
    node=Node()
    print('Current',k,' ',n)
    if k>=maxDepth:
        
        node = None
        
    else:
        q = random.randrange(0,col+1,1)
        node.k=k
        node.value=q
        node.n=n
        node.Left = CreateGrid(k+1)
        node.Right = CreateGrid(k+1)
    return node

a=CreateGrid(k1,0)
'''
def CreateGrid(category,k):
    node=Node()
    if k>=maxDepth:
        node = None
    else:
        q = random.randrange(0,col+1,1)
        if len(category[q])==0:
            node.Left=None
            node.Right=None
        else:
            Filter=(max(category[q])+min(category[q]))/2
            node.k=k
            node.value=category
            node.Left = CreateGrid(category.loc[category[q]<=Filter],k+1)
            node.Right = CreateGrid(category.loc[category[q]>Filter],k+1)
    return node

CreateGrid(category,k)