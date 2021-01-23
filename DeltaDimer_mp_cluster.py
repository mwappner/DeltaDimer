
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:32:07 2020

@author: Marcos
"""

from ddeint import ddeint
import numpy as np
import matplotlib.pyplot as plt
import warnings
from itertools import product
from multiprocessing import Pool, Lock, cpu_count
from utils import new_name, iter_to_csv

BASE_DIR = 'DeltaDimer_data/tauc_alpha/'
SAVEFILE = BASE_DIR + 'runs.txt'
SKIPFILE = BASE_DIR + 'skipped.txt'

WRITE_MODE = 'w' # 'w' or 'a'

CI = [1,5]

param_lists = {
    'd'    : 0.3,
    'tau'  : 3,
    'b_H'  : 20,
    'b_C'  : 20,
    'b_D'  : 30,
    'b_N'  : 30,
    'kCDp' : 0.1,
    'kCDm' : 0.1,
    'kCN'  : np.round(np.arange(0, 0.4, 0.05), 2),
    'kDN'  : np.round(np.arange(0, 0.4, 0.05), 2),
    'kEN'  : np.round(np.arange(0, 0.4, 0.05), 2),
    'H0'   : 1,
    'eta'  : 2.3,
    'a'    : [1, 5, 10],
    's'    : 1,
    'det'  : 1.0,
    'fc'   : 0
}

param_lists = {k:(v if hasattr(v, '__iter__') else [v]) for k, v in param_lists.items()} # make into lists
total_runs = 1
for l in param_lists.values():
    total_runs *= len(l)

parameters = product(*param_lists.values())

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
        -h1 + b_H * (1 + s*a * s1d**eta ) / ( (1 + s * s1d**eta) * (1 + h1d**eta) ),
        -c1 + b_C / (1 + h1c**eta) + lm * e1 - lp * c1 * d1 - kC * c1 * n2,
        -d1 + b_D + lm * e1 - lp * c1 * d1 - kD * d1 * n2,
        -e1 - lm * e1 + lp * c1 * d1 - kE * e1 * n2,
        -n1 + b_N - n1 * (kC * c2 + kD * d2 + kE * e2),
        -s1 +  n1 * (kC * c2 + kD * d2 + kE * e2),
        
        -h2 + b_H * (1 + s*a * s2d**eta ) / ( (1 + s * s2d**eta) * (1 + h2d**eta) ),
        -c2 + b_C / (1 + h2c**eta) + lm * e2 - lp * c2 * d2 - kC * c2 * n1,
        -d2 + b_D + lm * e2 - lp * c2 * d2 - kD * d2 * n1,
        -e2 - lm * e2 + lp * c2 * d2 - kE * e2 * n1,
        -n2 + b_N - n2 * (kC * c1 + kD * d1 + kE * e1),
        -s2 +  n2 * (kC * c1 + kD * d1 + kE * e1),
        ]
    return np.array(model)

def past_values(t):
    return np.concatenate( ( CI[0]*np.ones(6), CI[1]*np.ones(6) ) )

def print_asynch(lock, msg):
    lock.acquire()
    print(msg)
    lock.release()
    

def run_one(*parameters):
    
    tau = parameters[1]
    kCN, kDN, kEN = parameters[8:11]
    a, s, det, fc = parameters[-4:]
    
    sdet = f"detuning={det}"
    sfc = r'$\tau_c$=$\tau$*{:.2f}'.format(fc)
    name = f"{kCN=}; {kDN=}; {kEN=}; {a=}; {tau=}\n {sdet}; {sfc}; {CI=}"
    file_name = name.replace('\n', ';').replace('\\', '').replace('$','')
    
    parameters = make_parameters(*parameters)
    
    estimated_period = 2 * (tau + 1) # tau is automatically adimentionalized
    n = 1500
    target_n = 300 # approximate ammount of points per estimated period to save
    K = 60 # ammount of estiamted periods to integrate over
    
    print_asynch(lock, f'Starting {file_name}\n')
    
    try:
        tf = K * estimated_period
        N = n * K
        times = np.linspace(0, tf, N)
        
        Xint = ddeint(model_adim, past_values, times, fargs=parameters.values())
    except (UserWarning,RuntimeWarning) as w:
        
        print_asynch(lock, f"Couldn't complete {file_name} due to {w}")
        lock.acquire()
        with open(SKIPFILE, 'a') as skip:
            skip.write(file_name + '\n')
        lock.release()
        return
    
    lock.acquire()
    with open(SAVEFILE, 'a') as runs:
        runs.write(file_name + 
                   ',' + 
                   iter_to_csv(parameters.values()) + 
                   f',{CI[0]},{CI[1]}\n'
                   )
    lock.release()
    
    npy_name= new_name(BASE_DIR + file_name + '.npy')
    step = int(np.floor(n/target_n))
    Xsave = np.concatenate((np.expand_dims(times, 1), Xint), axis=1)
    np.save(npy_name, Xsave[::step])



def init(l):
    global lock
    lock = l
    
def main():    
    with warnings.catch_warnings():
        # warnings.filterwarnings("error", message='vode:', category=UserWarning)
        warnings.filterwarnings("error", message='overflow', category=RuntimeWarning)
        warnings.filterwarnings("error", message='divide by zero', category=RuntimeWarning)
        warnings.filterwarnings("error", message='invalid value', category=RuntimeWarning)

    
        print(f'Initializing run for {total_runs} values') 
        
        if WRITE_MODE == 'w':
            with open(SAVEFILE, 'w') as f:
                f.write('name,')
                f.write( iter_to_csv(make_parameters(*[1]*17).keys(), fmt='s'))
                f.write(',CI1,CI2\n')
        
        l = Lock()
        
        #fire off workers
        with Pool(cpu_count() + 2, initializer=init, initargs=(l,)) as pool:
        
            jobs = []
            for param_tuple in parameters:
                job = pool.apply_async(run_one, param_tuple )
                jobs.append(job)
    
            for job in jobs: 
                job.wait() # wait for all jobs to be done (blocks)

#%%
if __name__ == '__main__':
    from time import time
    start = time()
    main()    
    print(f'Done after {(time()-start)//60:.0f} minutes')