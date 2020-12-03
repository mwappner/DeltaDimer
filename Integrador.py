# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:21:40 2020

@author: Marcos
"""

from collections import deque
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


#%%

Model = Callable[[float, float, float], float]

class History(deque):
    '''A deque wrapper that forces a max length, initializes it with a default 
    value or ceros and adds one method to remove the oldest element and add a 
    new one.'''
    
    def __init__(self, length, defaut=0):
        if not isinstance(length, int):
            raise TypeError('length should be an integer.')
        super().__init__([defaut]*length, maxlen = length)
        
    def new_element(self, element):
        old = self.popleft()
        self.append(element)
        return old

def euler(model : Model, history, dt, T):
    output = []
    history = deque(history.copy()) # length of history is asumed to be the time delay
    
    # initial values
    ti = 0
    yi = history[-1]
    
    while ti < T:
        y_ip1 = yi + model(ti, yi, history.popleft()) * dt
        yi = y_ip1
        ti += dt
        output.append(y_ip1)
        history.append(y_ip1)
    
    return output


def rk2(model, history, dt, T, tau):
    ### midpoint rule
    
    n = int(tau/dt) # tau in units of dt aka how many time steps long the delay is
    
    # check I was given enough history
    assert len(history) >= n*2 + 1
    
    output = []
    
    # initial values
    ti = 0
    yi = history[-1]
    
    history = deque(history[-2*n-1:-1])
    
    # some constants
    dto2 = dt/2
    dtn = n*dt
    
    while ti < T:
        
        y1 = history[-n]
        y2 = history.popleft()
        
        k11 = model(ti, yi, y1)
        k12 = model(ti-dtn, y1, y2)
                
        k21 = model(ti+dto2, yi+dto2*k11, y1+dto2*k12)
        
        yi += dt * k21
        ti += dt
        
        output.append(yi)
        history.append(yi)      
    
    return output



def rk3(model, history, dt, T, tau):
    
    n = int(tau/dt) # tau in units of dt aka how many time steps long the delay is
    
    # check I was given enough history
    assert len(history) >= n*3 + 1
    
    output = []
    
    # initial values
    ti = 0
    yi = history[-1]
    
    history = deque(history[-3*n-1:-1])
    
    # some constants
    dto6 = dt/6
    dto2 = dt/2
    
    dtn = n*dt
    dt2n = 2*n*dt
    
    while ti < T:
        
        y1 = history[-n]
        y2 = history[-2*n]
        y3 = history.popleft()
        
        k11 = model(ti, yi, y1)
        k12 = model(ti-dtn, y1, y2)
        k13 = model(ti-dt2n, y2, y3)
        
        k21 = model(ti+dto2, yi+dto2*k11, y1+dto2*k12)
        k22 = model(ti+dto2-dtn, y1+dto2*k12, y2+dto2*k13)
        
        k31 = model(ti+dt, yi - dt*k11 + 2*dt*k21, y1 - dt*k12 + 2*dt*k22)
        
        yi += dto6 * (k11 + 4*k21 + k31)
        ti += dt
        
        output.append(yi)
        history.append(yi)     
    
    return output



def rk4(model, history, dt, T, tau):
        
    n = int(tau/dt) # tau in units of dt aka how many time steps long the delay is
    
    # check I was given enough history
    assert len(history) >= n*4 + 1
    
    output = []
    
    # initial values
    ti = 0
    yi = history[-1]
    
    history = deque(history[-4*n-1:-1])

    # some constants
    dto6 = dt/6
    dto2 = dt/2
    
    dtn = n*dt
    dt2n = 2*n*dt
    dt3n = 3*n*dt
    
    while ti < T:
        
        y1 = history[-n]
        y2 = history[-2*n]
        y3 = history[-3*n]
        y4 = history.popleft()
        
        k11 = model(ti, yi, y1)
        k12 = model(ti-dtn, y1, y2)
        k13 = model(ti-dt2n, y2, y3)
        k14 = model(ti-dt3n, y3, y4)
        
        k21 = model(ti+dto2, yi + dto2 * k11, y1 + dto2 * k12)
        k22 = model(ti+dto2-dtn, y1 + dto2 * k12, y2 + dto2 * k13)
        k23 = model(ti+dto2-dt2n, y2 + dto2 * k13, y3 + dto2 * k14)
        
        k31 = model(ti+dto2, yi + dto2 * k21, y1 + dto2 * k22)
        k32 = model(ti+dto2-dtn, y1 + dto2 * k22, y2 + dto2 * k23)
        
        k41 = model(ti+dt, yi + dt * k31, y1 + dt * k32)
        
        y_ip1 = yi + dto6 * (k11 + 2*k21 + 2*k31 + k41)
        yi = y_ip1
        ti += dt
        
        output.append(y_ip1)
        history.append(y_ip1)
    
    return output

#%% integrate and plot a sine

dt = 0.01 # time step
tau = 3*np.pi/2 # delay

T=40 # Total time

def function(t, y, yd):
    return yd

history_e = np.sin(np.arange(-tau, 0, dt))

data_e = euler(function, history_e, dt, T)
data_e = list(history_e) + data_e

history_r = np.sin(np.arange(-tau*4, 0, dt))
data_r2 = rk2(function, history_r, dt, T, tau)
data_r3 = rk3(function, history_r, dt, T, tau)
data_r4 = rk4(function, history_r, dt, T, tau)

t_e = np.linspace(-tau, T, len(data_e))
t_r = np.linspace(0, T, len(data_r4))

plt.plot(t_r, np.sin(t_r), 'k')
plt.plot(t_e, data_e, label='euler')
plt.plot(t_r, data_r2, '--', label='rk2')
plt.plot(t_r, data_r3, '-.', label='rk3')
plt.plot(t_r, data_r4, '--', label='rk4')
plt.legend()


#%% integrate and plot a circle

dt = 0.01 # time step
tau = 3*np.pi/2 # delay

T=40 # Total time

def function(t, y, yd):
    return yd

# Euler
history_e = np.array([
    np.cos(np.arange(-tau, 0, dt)), 
    np.sin(np.arange(-tau, 0, dt))
    ]).transpose()

data_e = euler(function, history_e, dt, T)
data_e = list(history_e) + data_e

# RK4
history_r = np.array([
    np.cos(np.arange(-tau*4, 0, dt)), 
    np.sin(np.arange(-tau*4, 0, dt))
    ]).transpose()

data_r4 = rk4(function, history_r, dt, T, tau)

data_e = np.array(data_e)
data_r = np.array(data_r4)

# t_e = np.linspace(-tau, T, len(data_e))
# t_r = np.linspace(0, T, len(data_r4))

# plt.plot(t_r, np.sin(t_r), 'k')
# plt.plot(t_e, data_e, '-', label='euler')
# plt.plot(t_r, data_r4, '--', label='rk4')
# plt.ylim([-5, 5])
plt.plot(data_e[:,0], data_e[:,1], label='euler')
plt.plot(data_r[:,0], data_r[:,1], label='rk4')
plt.legend()