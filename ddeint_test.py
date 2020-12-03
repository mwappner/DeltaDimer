# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 00:18:17 2020

@author: Marcos
"""

from ddeint import ddeint
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy import cos, sin, linspace, pi, array

# We solve the following system:
# Y(t) = 1 for t < 0
# dY/dt = -Y(t - 3cos(t)**2) for t > 0

def values_before_zero(t):
    return 1

def model(Y, t):
    return -Y(t - 3 * cos(Y(t)) ** 2)

tt = linspace(0, 30, 2000)
yy = ddeint(model, values_before_zero, tt)

fig, ax = plt.subplots(1, figsize=(4, 4))
ax.plot(tt, yy)

#%% Testing my sine

tau = 3*pi/2

def values_before_zero(t):
    return sin(t)

def model(Y, t):
    return Y(t-tau)

tt = linspace(0, 50, 2000)
yy = ddeint(model, values_before_zero, tt)

fig, ax = plt.subplots(1, figsize=(4, 4))
ax.plot(tt, yy)

#%% Github 2D test


def model(Y, t, d):
    x, y = Y(t)
    xd, yd = Y(t - d)
    return array([0.5 * x * (1 - yd), -0.5 * y * (1 - xd)])


g = lambda t: array([1, 2]) #values before zero
tt = linspace(2, 30, 20000) #time vector to evaluate

fig, ax = plt.subplots(1, figsize=(4, 4))

for d in [0, 0.2]:
    print("Computing for d=%.02f" % d)
    yy = ddeint(model, g, tt, fargs=(d,))
    # WE PLOT X AGAINST Y
    ax.plot(yy[:, 0], yy[:, 1], lw=2, label="delay = %.01f" % d)
    
#%% My circle test

tau = 3*pi/2

def values_before_zero(t):
    return array([cos(t), sin(t)])

def model(Y, t):
    return Y(t-tau)

tt = linspace(0, 50, 2000)
yy = ddeint(model, values_before_zero, tt)


fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 2)

axbig = fig.add_subplot(gs[:, 0])
axbig.plot(yy[:, 0], yy[:, 1])

axcos = fig.add_subplot(gs[0, 1])
axcos.plot(tt, yy[:,0])
axcos.set_title('cosine')

axsin = fig.add_subplot(gs[1, 1])
axsin.plot(tt, yy[:,1])
axsin.set_title('sine')