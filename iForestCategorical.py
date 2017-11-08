#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:26:11 2017

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
data = pd.read_csv('individualAnomaly.csv')


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

clf = IsolationForest(max_samples=length, random_state=rng)
clf.fit(category)
y_pred = clf.predict(category)
b=clf.decision_function(category)
for i in range(0,length):
    if b[i]<-0.1:
        print 'Anomaly Detected at:', i

'''
length=len(data)
col=16
rng = np.random.RandomState(42)

np.random.seed(42)
mu,sigma=0,0.00001
sa=np.random.normal(mu,sigma,length)
sb=np.random.normal(mu,sigma,length)
# Generate train data
X = np.vstack((data['client_lat'],data['client_lon']))

X = X.T
for i in range(0,length):
    if sp.isnan(X[i][0]):
        X[i][0]=200
for i in range(0,length):
    if sp.isnan(X[i][1]):
        X[i][1]=200 
X[:,0]=X[:,0]+sa
X[:,1]=X[:,1]+sb
clf = IsolationForest(max_samples=length, random_state=rng)
clf.fit(X)
y_pred = clf.predict(X)
b=clf.decision_function(X)


for i in range(0,length):
    if b[i]<-0.1:
        print 'Anomaly Detected at:', i
a=np.arange(length)



def siftdown(ary,e,begin,end):
    i,j=begin,begin*2+1
    while j<end:
        if j+1<end and ary[j+1]<ary[j]:
            j+=1
        if e<ary[j]:
            break
        ary[i]=ary[j]
        i,j=j,j*2+1
    ary[i]=e
    end = len(ary)
def heap_sort(ary,num):
    for i in range(end//2-1,-1,-1): 
        siftdown(ary,ary[i],i,end)
    for i in range(end-1,-1,-1):
        e=ary[i]
        ary[i]=ary[0]
        siftdown(ary,e,0,i)
    return ary[:-num-1:-1]

pre=np.zeros(length)

for i in range(0,length):
    pre[i]=predict[i][1]*predict[i][2]*predict[i][3]*predict[i][4]*predict[i][5]*predict[i][6]*predict[i][7]*predict[i][8]*predict[i][10]
    if pre[i]<10**-20:
        print 'Anomaly Detected at:', i
        
        
        
        
     
x=np.arange(length)
plt.figure(1)
plt.plot(x,pre)
'''
    
'''

x=np.arange(length)
preanom=heap_sort(pre,500)
plt.plot(x,pre)



def loginTimeCheck(self):
def loginFrequencyCheck(self):
def connectionTypeCheck(self):
def connection 

    
    loginCount={}
    for login in data['STATUS']:
        loginCount[login]=loginCount.get(login,0.0)+1.0
    return loginCount

    success_status=data['STATUS']
    psuccess_status=data['STATUS']
    for col in success_status:
        if col == "SUCCESS":
            psuccess_status[col] = 1
        else:
            psuccess_status[col] = 0
class NaiveBayes():
    """Naive Bayes Classifier class
    Implements the methods:
        CSV Read    - reads a data file
        Train       - Trains on a set of messages
        Feature_class_mean_sd - Calculates mean and sd
                    for FEATURE when CLASS = SPAM CLASS
        Classify    - Classifies a message
        P_spam_not_spam - Calculates probabilities a message
                        is spam or not spam
        Classification_test - tests if a message is correctly
                        classified
        Stratification_test - Performs 10-fold cross validation"""
        
    def __init__(self, corpus):
        self.type = corpus # Type of corpus - body or subject
        self.corpus_header, self.corpus_data = self.csv_read(corpus)
        self.corpus_data = self.cosine_normalisation()
        # Reads the corpus data
    
    def csv_read(self, corpus):
        """Reads a CSV file.  Outputs two lists: 
            corpus_float_data   - a list of messages
            corpus_header       - a list of headers"""
        corpus_data = []
        corpus_file = self.type + "loginAnomaly.csv" 
        reader = csv.reader(open(corpus_file)) 
        for row in reader:
            corpus_data.append(row) 
        corpus_header = corpus_data[:1] 
        corpus_data = corpus_data[1:]   
        corpus_float_data = [] 
        for row in corpus_data:
            float_row = [float(i) for i in row[:-1]]
            float_row.append(row[-1])
            corpus_float_data.append(float_row)
        return corpus_header, corpus_float_data



    def cosine_normalisation(self):
        """Performs the cosine normalisation of data"""
        self.normalised_data = []
        for message in self.corpus_data:
            normalised_scores = []
            tf_idf_scores = message[:-1]
            normalisation_factor = math.sqrt(sum([i**2 for i in tf_idf_scores])) 
            # Calculate \sum_{k} tf-idf(t_k, d_j)^2
            if normalisation_factor == 0:
                # Prevents dividing by zero
                self.normalised_data.append(message)
            else:       
                for score in tf_idf_scores:
                    normalised_scores.append(score/float(normalisation_factor))
                normalised_scores.append(message[-1])
                self.normalised_data.append(normalised_scores)
        return self.normalised_data
        
        
    def train(self, training_set):
        """Trains the classifier by calculating the prior normal distribution
        parameters for the feature sets and TRUE/FALSE"""
        training_messages = [self.corpus_data[i] for i in training_set] 
        # The set of training messages
        
        self.mean_sd_data = {}
        # Empty dictionary to hold mean and sd data
        
        for feature in range(200):
            self.mean_sd_data[feature] = {"Not Spam":[0, 0], "Spam":[0, 0]}
            for spam_class in ["Not Spam", "Spam"]:
                self.mean_sd_data[feature][spam_class] = []
            # Initialise the dictionary
            
        for feature in range(200):
            for spam_class in ["Not Spam", "Spam"]:
                # Fill the dictionary with values calculated from the feature_class_mean_sd method
                self.mean_sd_data[feature][spam_class] = self.feature_class_mean_sd(spam_class, feature, training_messages)
                
                
        # Calculate the a-priori spam and not-spam probabilities
        spam_count = 0
        for message in training_messages:
            if message[-1] == "Spam":
                spam_count += 1
        
        self.mean_sd_data["Spam"] = spam_count / float(len(training_set))
        self.mean_sd_data["Not Spam"] = 1 - (spam_count / float(len(training_set)))

        
        
    def feature_class_mean_sd(self, spam_class, feature, training_messages):
        """Calculates the mean and standard deviations for:
            FEATURE when CLASS = SPAM CLASS"""
        feature_list = []
        for message in training_messages:
            # Loop through all messages
            if spam_class == message[-1]:
                # If our message is in the right class
                feature_list.append(message[feature])
                # Take of the corresponding feature TF-IDF score
        # Return the summary statistics of the relevant feature / class
        return [mean(feature_list), sd(feature_list)]
        
        
    def classify(self, message):
        """Classify a message as spam or not spam"""
        p_spam = self.bayes_probability(message, "Spam") 
        # Probability that message is spam
        p_not_spam = self.bayes_probability(message, "Not Spam")
        # Probability that message is not spam
        # print p_spam, p_not_spam
        
        if p_spam > p_not_spam:
            return "Spam"
            # Message is not spam
        else:
            return "Not Spam"
            
            # Message is spam
            
            
    def bayes_probability(self, message, spam_class):
        """Probability that a message is or is not spam"""
            
        a_priori_class_probability = self.mean_sd_data[spam_class]
        # Probability that a single message is spam or not spam i.e. P(spam_id)
        # print "Commencing Bayes Probability on Message 0"
        # print "A priori Class Probability of {0} class is {1}".format(spam_class, a_priori_class_probability)
        class_bayes_probability = a_priori_class_probability


        body_best_features = [ 6,8,11,34,35,45,48,117,124,134,141,174] 
        # Feature selection from WEKA

        subject_best_features = range(1,200)
        
        
        if self.type == "body":
            """Converts the features f1, f2, ...fn into Python list indices"""
            best_features = map(lambda x :x -1, body_best_features)
        else:
            best_features = map(lambda x :x - 1, subject_best_features)
        for feature in best_features:
            # For all features
        
            message_tf_idf_score = message[feature]
            # Message tf_idf value
        
            tf_idf_mean = self.mean_sd_data[feature][spam_class][0]
            tf_idf_sd = self.mean_sd_data[feature][spam_class][1]
            # Get the parameters of the probability distribution governing this class
            
            prob_feature_given_class = norm_dist(message_tf_idf_score, tf_idf_mean, tf_idf_sd)
            # Find the probabilty P(tf-idf_feature = score | msg_class = class)
            class_bayes_probability = class_bayes_probability * prob_feature_given_class
            # Multiply together to obtain total probabilitiy
            # as per the Naive Bayes independence assumption

        return class_bayes_probability # Our probability that a message is spam or not spam
        
    def classification_test(self, message):
        """Tests if a message is correctly classified"""
        if self.classify(message) == message[-1]:
            return True
        else:
            return False
    
    def stratification_test(self):
        """Performs 10-fold stratified cross validation"""
        already_tested = []
        test_set  = []
        for i in range(10):
            """Create the set of 10 sixty element random bins"""
            sample = random.sample([i for i in range(600) if i not in already_tested], 60)
            already_tested.extend(sample)
            test_set.append(sample)
            
            
        results = []
        for validation_data in test_set:
            """Create the training set (540 elements) and the validation data (60 elements)"""
            training_sets = [training_set for training_set in test_set if training_set is not validation_data]
            training_data = []
            for training_set in training_sets:
                training_data.extend(training_set)
                
            self.train(training_data)
            # Train the probabilities of the Bayes Filter
            
            count = 0
            for index in validation_data:
                """Calculate the percentage of successful classifications"""
                if self.classification_test(self.corpus_data[index]):
                    count += 1
            results.append(float(count)/len(validation_data))
        return results  
            
#------------------------------------------------------------------------------


def print_results(results):
    """Formats results and prints them, along with summary statistic"""
    for result, index in zip(results, range(len(results))):
        print "Stratification Set {0} \t {1:.1f}% classified correctly.".format(index+1, result*100)
    print "##"*30
    print "\n\tOverall Accuracy is {0:.1f}%".format(mean(results) * 100)

if __name__ == '__main__':
    import random
    random.seed(18)
    #  Sets the seed, for result reproducibility
    test = NaiveBayes("subject")
    print "\tTesting Subject Corpus"
    print "##"*30
    results = test.stratification_test()
    print_results(results)
    print 
    print "\tTesting Body Corpus"
    print "##"*30
    test = NaiveBayesClassifier("body")
    results = test.stratification_test()
    print_results(results)
'''
