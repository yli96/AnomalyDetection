#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:31:40 2017

@author: yueningli
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
oneMiniteWindow=np.zeros(24*60)
oneMiniteDuplWindow=np.zeros(24*60*2)
fiveMinutesWindow=np.zeros(24*60/5)
fiveMinutesSlidingWindow=np.zeros(24*60-4)
tenMinutesWindow=np.zeros(24*60/10)
tenMinutesSlidingWindow=np.zeros(24*60-9)
data = pd.read_csv('shanghai.csv')
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


''''''

kde = KernelDensity(bandwidth=40.0, kernel='tophat')
kde.fit(minutereg[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(minutereg, np.full_like(minutereg, -0.01), '|k', markeredgewidth=1)
plt.ylim(0, 0.002)
''''''
kde = KernelDensity(bandwidth=40.0, kernel='epanechnikov')
kde.fit(minutereg[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(minutereg, np.full_like(minutereg, -0.01), '|k', markeredgewidth=1)
plt.ylim(0, 0.002)
''''''
kde = KernelDensity(bandwidth=40.0, kernel='exponential')
kde.fit(minutereg[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(minutereg, np.full_like(minutereg, -0.01), '|k', markeredgewidth=1)
plt.ylim(0, 0.002)
''''''
kde = KernelDensity(bandwidth=40.0, kernel='linear')
kde.fit(minutereg[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(minutereg, np.full_like(minutereg, -0.01), '|k', markeredgewidth=1)
plt.ylim(0, 0.002)
kde = KernelDensity(bandwidth=40.0, kernel='cosine')
kde.fit(minutereg[:, None])
'''
# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(minutereg, np.full_like(minutereg, -0.01), '|k', markeredgewidth=1)
plt.ylim(0, 0.002)
hist = plt.hist(minutereg, bins=80, normed=True)
density, bins, patches = hist
widths = bins[1:] - bins[:-1]
'''
'''
hist = plt.hist(minutereg, bins=80, normed=True)
density, bins, patches = hist
widths = bins[1:] - bins[:-1]
kde = KernelDensity(bandwidth=20.0, kernel='gaussian')
kde.fit(minutereg[:, None])
x_d = np.linspace(0, 1440, 1440)
# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

density = sum(norm(xi).pdf(x_d) for xi in minutereg)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(minutereg, np.full_like(minutereg, -0.1), '|k', markeredgewidth=1)

plt.axis([0, 1440, -0.2, 200]);


plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(minutereg, np.full_like(x, 10), '|k', markeredgewidth=1)
plt.ylim(0, 0.003)

from sklearn.neighbors import KernelDensity

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=20.0, kernel='gaussian')
kde.fit(minutereg[:, None])
x_d = np.linspace(0, 1440, 1440)

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=20)
plt.plot(minutereg, np.full_like(minutereg, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)

from scipy.stats import norm
x_d = np.linspace(0, 1440, 1440)
density = sum(norm(xi).pdf(x_d) for xi in minutereg)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(k, np.full_like(k, -0.1), '|k', markeredgewidth=10)

plt.axis([-100, 1500, -0.2, 200]);
'''
'''
oneMiniteWindow=np.zeros(24*60)
fiveMinutesWindow=np.zeros(24*60/5)
fiveMinutesSlidingWindow=np.zeros(24*60-4)
tenMinutesWindow=np.zeros(24*60/10)
tenMinutesSlidingWindow=np.zeros(24*60-9)

col=16
c=0
logtime=data['_time']
for i in range(0,length): 
    k[i]=int(str(logtime[i][11:13]+''+logtime[i][14:16]))
    a=k[i]
    q[a]=q[a]+1
for i in range(0,length):
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

'''

'''    


X=k
X=X.reshape(-1,1)
X_plot=np.linspace(0, 2400, 2400)[:, np.newaxis]
hist = plt.hist(k, bins=20, normed=True)

    
#----------------------------------------------------------------------
# Plot the progression of histograms to kernels
np.random.seed(1)

X_plot=np.linspace(0, 2400, 1)[:, np.newaxis]


fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# histogram 1
ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
ax[0, 0].text(-3.5, 0.31, "Histogram")

# histogram 2
ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")

# tophat KDE
kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")

# Gaussian KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")

for axi in ax.ravel():
    axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)

for axi in ax[:, 0]:
    axi.set_ylabel('Normalized Density')

for axi in ax[1, :]:
    axi.set_xlabel('x')

#----------------------------------------------------------------------
# Plot all available kernels
X_plot = np.linspace(-6, 6, 1000)[:, None]
X_src = np.zeros((1, 1))

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)


def format_func(x, loc):
    if x == 0:
        return '0'
    elif x == 1:
        return 'h'
    elif x == -1:
        return '-h'
    else:
        return '%ih' % x

for i, kernel in enumerate(['gaussian', 'tophat', 'epanechnikov',
                            'exponential', 'linear', 'cosine']):
    axi = ax.ravel()[i]
    log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(X_plot)
    axi.fill(X_plot[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
    axi.text(-2.6, 0.95, kernel)

    axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axi.xaxis.set_major_locator(plt.MultipleLocator(1))
    axi.yaxis.set_major_locator(plt.NullLocator())

    axi.set_ylim(0, 1.05)
    axi.set_xlim(-2.9, 2.9)

ax[0, 1].set_title('Available Kernels')

#----------------------------------------------------------------------
# Plot a 1D density example
N = 100
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')

for kernel in ['gaussian', 'tophat', 'epanechnikov']:
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
            label="kernel = '{0}'".format(kernel))

ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.show()
'''