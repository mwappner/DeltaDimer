# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:47:25 2021

@author: Marcos
"""

from ddeint import ddeint
import numpy as np
# import matplotlib.pyplot as plt
from itertools import product
from scipy.signal import hilbert
import warnings
import os
from utils import iter_to_csv, new_name, Testimado, make_dirs_noreplace
import sys

BASE_DIR = 'DeltaDimer_data'

run_number = 7

args = sys.argv
if len(args)>1:
    run_number = int(args[1])
    
BASE_DIR = make_dirs_noreplace(os.path.join(BASE_DIR, str(run_number)))
SAVEFILE = os.path.join(BASE_DIR, 'runs.csv')
SKIPFILE = os.path.join(BASE_DIR, 'skipped.csv')

def model_adim(X, t, tau, 
               b_H, b_C, b_D, b_N, # b == beta
               eta, a, s, # a == alpha, s == sigma
               lm, lp, kC, kD, kE, #l == lambda, k == kappa, lm == lm/H0
               det, fc): #det == detuning , fc*tau == tauc
    
    h1, c1, d1, e1, n1, s1, h2, c2, d2, e2, n2, s2 = X(t)
    h1d, _, _, _, _, s1d, _, _, _, _, _, _ = X(t-tau)
    _, _, _, _, _, _, h2d, _, _, _, _, s2d = X(t-tau*det)
    h1c, _, _, _, _, _, h2c, _, _, _, _, _ = X(t-tau*fc)
    #_, _, _, _, _, _, h2d, _, _, _, _, s2d = X(t-tau*detune)
    
    model = [
        -h1 + b_H * (1 + a * (s * s1d)**eta ) / ( (1 + (s * s1d)**eta) * (1 + h1d**eta) ),
        -c1 + b_C / (1 + h1c**eta) + lm * e1 - lp * c1 * d1 - kC * c1 * n2,
        -d1 + b_D + lm * e1 - lp * c1 * d1 - kD * d1 * n2,
        -e1 - lm * e1 + lp * c1 * d1 - kE * e1 * n2,
        -n1 + b_N - n1 * (kC * c2 + kD * d2 + kE * e2),
        -s1 +  n1 * (kC * c2 + kD * d2 + kE * e2),
        
        -h2 + b_H * (1 + a * (s* s2d)**eta ) / ( (1 + (s * s2d)**eta) * (1 + h2d**eta) ),
        -c2 + b_C / (1 + h2c**eta) + lm * e2 - lp * c2 * d2 - kC * c2 * n1,
        -d2 + b_D + lm * e2 - lp * c2 * d2 - kD * d2 * n1,
        -e2 - lm * e2 + lp * c2 * d2 - kE * e2 * n1,
        -n2 + b_N - n2 * (kC * c1 + kD * d1 + kE * e1),
        -s2 +  n2 * (kC * c1 + kD * d1 + kE * e1),
        ]
    return np.array(model)

def make_parameters(d, tau, b_H, b_C, b_D, b_N, kCDp, kCDm, kCN, kDN, kEN, H0, eta, a, s, det, fc):
    parameters = {
        'tau' : tau       ,
        'b_H' : b_H/d/H0  ,
        'b_C' : b_C/d/H0  ,
        'b_D' : b_D/d/H0  ,
        'b_N' : b_N/d/H0  ,
        'eta' : eta       , # hill exponent
        'a'   : a         , # alpha
        's'   : s         , # sigma
        'lm'  : kCDm/d    , # lm == lm/H0
        'lp'  : kCDp*H0/d ,
        'kC'  : kCN*H0/d  ,
        'kD'  : kDN*H0/d  ,
        'kE'  : kEN*H0/d  ,
        'det' : det       ,
        'fc'  : fc        ,
        }
    return parameters


with open(SAVEFILE, 'w') as f:
    f.write('name,')
    f.write( iter_to_csv(make_parameters(*[1]*17).keys(), fmt='s'))
    f.write(',CI1,CI2,mean_phase,trend\n')


CI = [0, 1]

param_lists = {
    'd'    : 0.3,
    'tau'  : 3,
    'b_H'  : 1,
    'b_C'  : [1,3,5,10],
    'b_D'  : [1,3],
    'b_N'  : [1,3,5],
    'kCDp' : 0,
    'kCDm' : 0,
    'kCN'  : np.round(np.logspace(-3, -0.7, 18), 6),
    'kDN'  : np.round(np.logspace(-5, -1, 5), 6),
    'kEN'  : 0,
    'H0'   : 1,
    'eta'  : 2.3,
    'a'    : 5,
    's'    : 0.1,
    'det'  : 1.0,
    'fc'   : 0
}

param_lists = {k:(v if hasattr(v, '__iter__') else [v]) for k, v in param_lists.items()} # make into lists
total_runs = 1
for l in param_lists.values():
    total_runs *= len(l)

parameters = product(*param_lists.values())

def find_point_by_value(array, value):
    return np.abs(array-value).argmin()

def past_values(t):
    return np.concatenate( ( CI[0]*np.ones(6), CI[1]*np.ones(6)) )

n = 1200 # ammount of points per estimated period
K = 80 # ammount of estiamted periods to integrate over
N = n * K
target_n = 300 # approximate ammount of points per estimated period to save
buffer_periods = 4 # ammount of periods at the end to descart during hilbert transform
target_periods_proportion =  0.1 # proportion of the total ammount of periods to calcualte the hilbert transform over
stationary_estimate_proportion = 0.6 # proportion of periods to count as stationary, estimate

i = 0
t_est = Testimado(total_runs)

with warnings.catch_warnings():
    # warnings.filterwarnings("error", message='vode:', category=UserWarning)
    warnings.filterwarnings("error", message='overflow', category=RuntimeWarning)
    warnings.filterwarnings("error", message='divide by zero', category=RuntimeWarning)
    warnings.filterwarnings("error", message='invalid value', category=RuntimeWarning)
    
    for p in parameters:
        
        i += 1
        if i%10==0:
            print(f'Running {i} ot of {total_runs}. ETA = {t_est.time_str(i)}')
        
        this_params = make_parameters(*p)
        tau = p[1]
        kCN, kDN, kEN = p[8:11]
        eta, a, s, det, fc = p[-5:]
        
        sdet = f"detuning={det}"
        sfc = r'tau_c={:.2f}'.format(fc)
        file_name = f"{kCN=}; {kDN=}; {kEN=}; {a=}; {tau=}; {s=}; {sdet}; {sfc}; {CI=}".replace(',', ';')
        file_name = new_name(os.path.join(BASE_DIR, file_name+'.npy'), newformater=' (%d)')
        file_name, _ = os.path.splitext(os.path.basename(file_name))
        print(file_name)
        
        
        estimated_period = 2 * (tau + 1)       
        tf = K * estimated_period
        times = np.linspace(0, tf, N)
        
        try:
            
            Xint = ddeint(model_adim, 
                      past_values, 
                      times, 
                      fargs=this_params.values())
            
            # hilbert transform to get phases
            points = int(stationary_estimate_proportion * N) # points in stationary region
            h1, h2 = Xint[:,0][-points:], Xint[:, 6][-points:]
            h1_phase = np.unwrap(np.angle(hilbert(h1 - h1.mean())))
            h2_phase = np.unwrap(np.angle(hilbert(h2 - h2.mean())))
            
            # get integer ammount of periods
            total_periods = int(np.floor(h1_phase.max()/(2*np.pi))) # should be close to K
            target_periods = int(total_periods * target_periods_proportion)
            last_full_period_value = total_periods * np.pi * 2
            start_inx = find_point_by_value(h1_phase, last_full_period_value - 2*np.pi*(buffer_periods+target_periods))
            end_inx = find_point_by_value(h1_phase, last_full_period_value - 2*np.pi*buffer_periods)
            
            mean_phase_diff = np.mean(phase_diff := (h1_phase[start_inx:end_inx] - h2_phase[start_inx:end_inx]))
        
            # trends
            points_per_period = int((end_inx - start_inx)/target_periods)
            moving_mean = np.array([phase_diff[i : points_per_period + i].mean() for i in range((target_periods-1) * points_per_period)])
            
            L = moving_mean.size
            polytimes = np.linspace(0, L * tf/N, L)
            lin = np.polyfit(polytimes, moving_mean, 1)

            npy_name = new_name(os.path.join(BASE_DIR, file_name + '.npy'))
            step = int(np.floor(n/target_n))
            Xsave = np.concatenate((np.expand_dims(times, 1), Xint), axis=1)
            np.save(npy_name, Xsave[::step])
            
            with open(SAVEFILE, 'a') as runs:
                runs.write(file_name + ',' + 
                   iter_to_csv(this_params.values(), '.3e') + 
                   f',{CI[0]},{CI[1]},{mean_phase_diff:.3e},{lin[0]:.3e}\n'
                       )

        except (UserWarning,RuntimeWarning):
            
            with open(SKIPFILE, 'a') as skip:
                skip.write(
                    iter_to_csv(this_params.values()) + 
                    f',{CI[0]},{CI[1]}\n'
                   )

# print('\a')
