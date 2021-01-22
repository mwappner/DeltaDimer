# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:30:09 2020

@author: Marcos
"""

# scanning over detailed b and d ranges
SAVEFILE = 'taud_scan.csv'
SKIPFILE = 'taud_skipped.csv'
WRITE_MODE = 'w' # 'a' or 'w'

from ddeint import ddeint as ddeint
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import warnings
from multiprocessing import Lock, Pool, cpu_count
from itertools import product


d_list = np.concatenate( (np.round(np.arange(0.001, 0.1, 0.01), 3),
                          np.round(np.arange(0.1, 1.2, 0.025), 3),
                          np.round(np.arange(1.2, 2.5, 0.05), 2),
                          )
                        )
b_list = [20, 35, 60]
h_list = [2, 3]
tau_list = range(25)
x0_list = [1, 2]

parameters = product(d_list, b_list, h_list, tau_list, x0_list)

total_runs = len(d_list) * len(b_list) * len(h_list) * len(tau_list) * len(x0_list)

N = 3000
plot_range = slice(-int(N/3), N) # equivalent to -10*estimated_period/dt


def convert(array_of_arrays):
    '''Regularizes format of the output so that they all are arrays of arrays'''
    if (test := array_of_arrays[0]).size == 1:
        array_of_arrays = np.fromiter((a for a in array_of_arrays), test.dtype, array_of_arrays.size)
    
    return array_of_arrays
    
    
def oscilates(x, minimal_size=0.05):
    '''Decides if x presents oscilations given a minimal amplitude'''
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
    '''Re adjust the limits of ax given the minimum and maximum of the plotted data,
    but ensure size is at most some minimal value'''
    
    whitespace = (max_data-min_data) * whitespace_rate
    y_min = min_data - whitespace
    y_max = max_data + whitespace
    
    if y_max - y_min < min_size:
        center = (y_min + y_max)/2
        ax.set_ylim([center-min_size/2, center+min_size/2])
    else:
        ax.set_ylim([y_min, y_max])


def model(x, t, d, b, x0, h, tau):
    return -d * x(t) + b / (1+(x(t-tau)/x0)**h)

def history(t):
    return 1.01

lock = Lock()

def run_one(d, b, h, tau, x0):
    
    estimated_period = 2*tau + 2/d
    tf = 30 * estimated_period   
    dt = tf/N
    times = np.linspace(0, tf, N)
    
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message='vode:', category=UserWarning)
        warnings.filterwarnings("error", message='DUODE--:', category=UserWarning)
        try:
            x = ddeint(model, history, times, fargs=[d, b, x0, h, tau])
            
            # ax1.plot(tt := times[plot_range], px := convert(x[plot_range]))
            # relim(ax1, px.max(), px.min())
            # ax2.plot(tt, px)
            # # relim(ax2, px.max(), px.min(), min_size=0)
            
            px = convert(x[plot_range])
            osc, amp, period, mins, maxs = oscilates(px)
            
            # ax2.plot(tt[mins], px[mins], 'o', color='C2')
            # ax2.plot(tt[maxs], px[maxs], 'o', color='C1')
            
            # plt.savefig(f'regulacion_imgs_mp/d={d}, b={b}, h={h}, tau={tau}, x0={x0}.png')

            lock.acquire()
            with open(SAVEFILE, 'a') as f:
                f.write(f'{d},{b},{h},{tau},{x0},{osc},{amp},{period*dt}\n')
            lock.release()
            
            
        except UserWarning as w:
            
            lock.acquire()
            with open(SKIPFILE, 'a') as skip:
                skip.write(f'{d},{b},{h},{tau},{x0}\n')
                # warnings.warn(w)
            lock.release()
            
        finally:
            pass
            # plt.close(fig)
            

def init(l):
    global lock
    lock = l
    

def main():  
        
    print(f'Initializing run for {total_runs} values') 
    
    if WRITE_MODE == 'w':
        with open(SAVEFILE, 'w') as f, open(SKIPFILE,'w') as skip:
            f.write('d,b,h,tau,x0,osc?,amp,period\n')
            skip.write('d,b,h,tau,x0\n')
        
    
    # lock = Manager().Lock()
    l = Lock()
    
    #fire off workers
    with Pool(cpu_count() + 2, initializer=init, initargs=(l,)) as pool:
    
        jobs = []
        for param_tuple in parameters:
            job = pool.apply_async(run_one, param_tuple )
            jobs.append(job)

        for job in jobs: 
            job.get() #wait for all jobs to be done (blocks)

if __name__ == '__main__':
    from time import time
    start = time()
    main()    
    print(f'Done after {time()-start}')