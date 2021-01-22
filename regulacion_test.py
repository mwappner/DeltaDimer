# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:20:57 2020

@author: Marcos
"""


from ddeint import ddeint as ddeint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import fft, signal
import warnings

def convert(array_of_arrays):
    if (test := array_of_arrays[0]).size == 1:
        array_of_arrays = np.fromiter((a for a in array_of_arrays), test.dtype, array_of_arrays.size)
    
    return array_of_arrays
    

#%% Tests

d = 0.3
b = 30
h = 2.3
tau = 10
x0 = 1

estimated_period = 2*tau + 2/d

tf = 30 * estimated_period
N = 2000
dt = tf/N
plot_range = slice(int(-10*estimated_period/dt), N)

def model(x, t, d, b, x0, h, tau):
    return -d * x(t) + b/(1+(x(t-tau)/x0)**h)

def history(t):
    return 1

reg_func = lambda x: model(lambda t: t, x, d, b, x0, h, tau) # pongo x(t) = t

times = np.linspace(0, tf, N)
x = ddeint(model, history, times, fargs=[d, b, x0, h, tau])

x_to_plot = np.linspace(0, 10, 100)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(15, 6))

ax1.plot(x_to_plot, reg_func(x_to_plot))
ax1.set_title('Function')
ax1.grid(True)

ax2.plot(times, x)
ax2.set_title('Time series')
ax2.grid(True)

x_end = convert(x[plot_range])
N_x = len(x_end)
tf_x = 10*estimated_period/dt

# real part of FFT without frequency 0
fx = 2/N_x * np.abs(fft.fft(x_end))[1:N_x//2]
ffreq = fft.fftfreq(N_x, tf_x/N_x)[1:N_x//2]
fpeaks = signal.find_peaks(fx, prominence = 0.05)[0]

ax3.plot(ffreq, fx, label = sum(fx)/ffreq[0])
ax3.set_title('Frequencies')
ax3.plot(ffreq[fpeaks], fx[fpeaks], 'ro')
# plt.legend()


c = np.correlate(x_end, x_end, 'full')
ax4.plot(c)
ax4.set_title('Correlation')

fc = np.abs(fft.fft(c)/N)[100:N_x]
cpeaks = signal.find_peaks(fc, prominence=10)[0]
ax5.plot(fc)
ax5.plot(cpeaks, fc[cpeaks], 'ro')

plt.figure()
plt.plot(times[plot_range], x[plot_range])

#%% funcs

# def oscilates(x, tf):
#     # correlate, fourier transform, find peaks
#     N = len(x)
    
#     c = np.correlate(x, x, 'full')
#     fc = np.abs(fft.fft(c)/N)[100:N]
#     cpeaks = signal.find_peaks(fc, prominence=10)[0]
    
#     if not cpeaks.size:
#         return False, None
#     else:
#         fx = 2/N * np.abs(fft.fft(x))[1:N//2]
#         ffreq = fft.fftfreq(N, tf/N)[1:N//2]
#         # overkill?
#         # fpeaks = signal.find_peaks(fx, prominence = 0.05)[0] 
        
#         max_freq_index = np.argmax(fx)
#         return True, ffreq[max_freq_index]
    
def oscilates(x, minimal_size=0.05):
    # options to better detect peaks:
        # distance = 0.5*estimated_period
        # prominence = 0.01
        # width = 3 ?
    max_ind = signal.find_peaks(x)[0]
    min_ind = signal.find_peaks(-x)[0]
    maxima = x[max_ind]
    minima = x[min_ind]
    
    if maxima.size > 0 and minima.size > 0:
        try:
            av_distance = np.average(maxima-minima)
        except ValueError: # arrays of different size
            # print('Different size!')
            av_distance = maxima.mean() - minima.mean()
    else:
        av_distance = np.nan
    
    estimated_period = np.diff(max_ind).mean() if max_ind.size>1 else np.nan
    
    
    return av_distance > minimal_size, av_distance, estimated_period, min_ind, max_ind
    
    
def relim(ax, max_data, min_data, min_size = 0.1, whitespace_rate=0.1):
    
    whitespace = (max_data-min_data) * whitespace_rate
    y_min = min_data - whitespace
    y_max = max_data + whitespace
    
    if y_max - y_min < min_size:
        center = (y_min + y_max)/2
        ax.set_ylim([center-min_size/2, center+min_size/2])
    else:
        ax.set_ylim([y_min, y_max])
#%% scan

from time import time

start = time()

d_list = np.round(np.arange(0.1, 0.5, 0.1), 3)
b_list = [2]
h_list = [2, 3]
tau_list = range(1, 10)
x0_list = [1]

total_runs = len(d_list) * len(b_list) * len(h_list) * len(tau_list)

N = 3000
plot_range = slice(-int(N/3), N) # equivalent to -10*estimated_period/dt


def model(x, t, d, b, x0, h, tau):
    return -d * x(t) + b / (1+(x(t-tau)/x0)**h)

def history(t):
    return 1


fig, (ax1, ax2) = plt.subplots(2, 1)
line1, = ax1.plot(list(range(int(N/3))))
line2, = ax2.plot(list(range(int(N/3))))

x0 = 1
runs = 1
with open('regulacion_imgs/regulacion.csv', 'a') as f, open('regulacion_imgs/skipped.csv', 'a') as skip:
    with warnings.catch_warnings() as w:
        warnings.simplefilter("error", category=UserWarning)
    
        #  f.write('d,b,h,tau,x0,osc?,amp,period\n')
        
        for d in d_list:
            for tau in tau_list:
                
                estimated_period = 2*tau + 2/d
    
                tf = 30 * estimated_period
                dt = tf/N
                # plot_range = slice(int(-10*estimated_period/dt), N)     
                
                
                times = np.linspace(0, tf, N)
                line1.set_xdata(tt := times[plot_range])
                ax1.set_xlim( (tt[0], tt[-1]) )
                line2.set_xdata(tt)
                ax2.set_xlim( (tt[0], tt[-1]) )
                
                for b in b_list:
                    for h in h_list: # Con h = 1 no conseuguí oscilaciones nunca
                        
                        print(f'Run #{runs} of {total_runs} with: d={d}, b={b}, h={h}, tau={tau}, x0={x0}')
                                            
                        try:
                            x = ddeint(model, history, times, fargs=[d, b, x0, h, tau])
                        except UserWarning:
                            skip.write(f'{d},{b},{h},{tau},{x0}\n')
                        
                        line1.set_ydata(px := convert(x[plot_range]))
                        relim(ax1, px.max(), px.min())
                        line2.set_ydata(px)
                        relim(ax2, px.max(), px.min(), min_size=0)
        
                        # name_maker = lambda l: ', '.join([f'{i}={{{i}}}' for i in l.split()])
                        
                        osc, amp, period, mins, maxs = oscilates(px)
                        
                        l2, = ax2.plot(tt[mins], px[mins], 'o', color='C2')
                        l3, = ax2.plot(tt[maxs], px[maxs], 'o', color='C1')
                        
                        plt.savefig(f'regulacion_imgs/d={d}, b={b}, h={h}, tau={tau}, x0={x0}.png')
                        
                        l2.remove()
                        l3.remove()
                        del l2, l3
                        
                        f.write(f'{d},{b},{h},{tau},{x0},{osc},{amp},{period*dt}\n')
                        
                        runs+=1

print(f'Done after {time()-start}')

#%% run single


def model(x, t, d, b, x0, h, tau):
    return -d * x(t) + b / (1.0+(x(t-tau)/x0)**h)

def history(t):
    return 1

#Excess work warning
d=1.125
b=1
h=3
tau=24
x0=1

# #Numpy warning
# d=0.3
# b=3
# h=3
# tau=10
# x0=1

estimated_period = 2*tau + 2/d

tf = 25 * estimated_period
N = 2000
dt = tf/N
plot_range = slice(int(-5*estimated_period/dt), N)

times = np.linspace(0, tf, N)
# fig, ax = plt.subplots()
# line, = ax.plot(times[1200: 1600], times[1200: 1600])

# print(f'Run #{runs} with: d={d}, b={b}, h={h}, tau={tau}, x0={x0}')

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
with warnings.catch_warnings() as w:
    warnings.filterwarnings("error", message='vode:', category=UserWarning)
    warnings.filterwarnings("error", message='DUODE--:', category=UserWarning)
    try:
        
        x = ddeint(model, history, times, fargs=[d, b, x0, h, tau])
    
    # line.set_ydata(px := x[: 1600])
    # relim(ax, px.max()[0], px.min()[0])
    
    # # name_maker = lambda l: ', '.join([f'{i}={{{i}}}' for i in l.split()])
    # plt.savefig(f'regulacion_imgs/d={d}, b={b}, h={h}, tau={tau}, x0={x0}.png')
    
    
    # with open('regulacion_imgs/regulacion.csv', 'a') as f:
    #     osc, freq = oscilates(x, N, tf)
    #     f.write(f'{d},{b},{h},{tau},{x0},{osc},{freq}\n')
    
        plt.plot(times[plot_range], x[plot_range])
    except UserWarning as w:
        warnings.warn(w)


