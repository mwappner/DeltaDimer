# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:53:26 2021

@author: Marcos
"""

import matplotlib.pyplot as plt
import numpy as np
import functools
from pandas import DataFrame as df

from utils import contenidos


def big_or(column, values):
    return functools.reduce(lambda x, y: x|y, (column==v for v in values))

def filter_data(df, filters):
    """ Takes in a dictionary with keys being the column name to check and 
    values being the values to match for that column, either floats or tuples,
    as well as the dataframe to filter. Returns the filtered dataframe."""

    # make floats into iterables
    filters = {k: (v if hasattr(v, '__iter__') else [v]) for k, v in filters.items()}
    
    # create bool masks
    bools = [big_or(df[k], v) for k, v in filters.items()]
    
    # make a big and
    where = functools.reduce(lambda x, y: x&y, bools)
    
    return df[where]


base_dir = 'DeltaDimer_data/no_tauc/'
INDEX = 171 #146, 171

files = contenidos(base_dir, filter_ext='.npy')

X = np.load(ff:=files[INDEX])[::10]
file_name = ff.split('/')[1][:-4]

with open(base_dir + 'runs.txt', 'r') as param_file:
    param_array = [l.strip() for l in param_file]
param_names = param_array[0].split(',')
param_values = [l.split(',') for l in param_array[1:]]
param_values = df(data=[[','.join(l[:3]), *l[3:]] for l in param_values],
                  columns=param_names)

names = ['H1', 'C1', 'D1', 'E1', 'N1', 'S1', 
         'H2', 'C2', 'D2', 'E2', 'N2', 'S2']

d=0.3
tau = 10
estimated_period = 2 * (d*tau + 1)
n = 400 # ammount of points per estimated period
K = 40 # ammount of estiamted periods to integrate over

tf = K * estimated_period
N = n * K
dt = tf/N
plot_range = slice(int(-K*estimated_period/dt), N)
times = np.linspace(0, tf, N)
tt = times[plot_range]

f, axarr = plt.subplots(2,3, figsize=(13,6))

row = param_values[param_values['name']==file_name]
f.suptitle(file_name)

for i, ax in enumerate(axarr.flatten()):
    name1, name2 = names[i], names[i+6]
    x1, x2 = X[:,i], X[:, i+6]
    ax.plot(tt, x1[plot_range], lw=1.5, label = name1 )
    ax.plot(tt, x2[plot_range], '--' , lw=1.5, label = name2 )
    
    ax.set_title(name1[0])
    plt.legend()