#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:53:32 2021

@author: user
"""


import numpy as np
import functools
import pandas as pd
import os

import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import interp1d

from utils import contenidos, find_numbers


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


def filter_single(df, param_dict, plot=False):
    
    data = filter_data(df, param_dict )
    
    if plot:
        I = plt.imread('regulacion_imgs/{}.png'.format(', '.join([f'{k}={v}' for k, v in param_dict.items()])))
        plt.imshow(I)
        plt.axis('off')
        plt.title('amp = {:.2e}'.format(data['amp'].values[0]))
    
    return data


def relim(ax, max_data, min_data, min_size = 0.1, whitespace_rate=0.1):
    
    whitespace = (max_data-min_data) * whitespace_rate
    y_min = min_data - whitespace
    y_max = max_data + whitespace
    
    if y_max - y_min < min_size:
        center = (y_min + y_max)/2
        ax.set_ylim([center-min_size/2, center+min_size/2])
    else:
        ax.set_ylim([y_min, y_max])
        
def make_parameters(d=0.3, tau=3, b_H=20, b_C=20, b_D=30, b_N=30, kCDp=0.1, kCDm=0.1, kCN=0, kDN=0, kEN=0, H0=1, eta=2.3, a=5, s=1, det=1, fc=0):
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

def one_by_name(df, name, image=False):
    return df.loc[name.replace(',', ';')]

def filter_by_values(df, kCN=None, kDN=None, kEN=None, a=None, tau=None, s=None, det=None, fc=None, CI=None):
    kC = [round(make_parameters(kCN=k)['kC'], 2) for k in (kCN if hasattr(kCN, '__iter__') else [kCN])] if kCN is not None else None
    kD = [round(make_parameters(kDN=k)['kD'], 2) for k in (kDN if hasattr(kDN, '__iter__') else [kDN])] if kDN is not None else None
    kE = [round(make_parameters(kEN=k)['kE'], 2) for k in (kEN if hasattr(kEN, '__iter__') else [kEN])] if kEN is not None else None
    filter_dict = dict(kC=kC, kD=kD, kE=kE, a=a, tau=tau, s=s, det=det, fc=fc, CI=CI)
    for k, v in list(filter_dict.items()):
        if v is None:
            filter_dict.pop(k)
            
    return filter_data(df, filter_dict)


def one_by_params(kCN, kDN, kEN, a, tau, det, fc, CI, image=False):
    
    sdet = f"detuning={det}"
    sfc = f'tau_c=tau*{fc:.2f}'
    name = f"kCN={float(kCN)}; kDN={float(kDN)}; kEN={float(kEN)}; {a=}; {tau=}; {sdet}; {sfc}; {CI=}"
    return name


def find_point_by_value(array, value):
    return np.abs(array-value).argmin()

#%% Load data

BASE_DIR = 'DeltaDimer_data/tauc_alpha/manual_runs/'

# current_dir = 6
# BASE_DIR = 'DeltaDimer_data/tauc_alpha/'
# BASE_DIR = os.path.join(BASE_DIR, str(current_dir))

files = contenidos(BASE_DIR, filter_ext='.npy', sort='age')
runs = pd.read_csv(os.path.join(BASE_DIR, 'runs.csv'), index_col='name')

parameters = runs.keys()
parameter_values = {k:sorted(list(set(runs[k]))) for k in parameters}


#%% Make orbits
runs = pd.read_csv(BASE_DIR + 'runs.csv', index_col='name')
files.update()


# =============================================================================
#### For manual runs
FILE_INDX = -1 # newest one
this_name, _ = os.path.splitext(os.path.basename(files[FILE_INDX]))
some_s = one_by_name(runs, this_name)

#get parameters
eta = some_s['eta']
alpha = some_s['a']
sigma = some_s['s']
tau = some_s['tau']
name = this_name
# =============================================================================

# get file and figure names
ci1, ci2 = int(some_s["CI1"]), int(some_s["CI2"])

# file = BASE_DIR + name.replace(f'CI=[{ci1}; {ci2}]', f'CI=[{ci1}, {ci2}]') + '.npy'
file = BASE_DIR + name + '.npy'
numbers = find_numbers(name)
fig_name = 'kCN={}, kDN={}, kEN={}, a={}, $\\tau$={}, s={}\ndetuning={}, $\\tau_c$=$\\tau$*{}, CI=[{},{}]'.format(*numbers)
fig_name += r', $\eta$={}'.format(some_s["eta"])

# load and unpack data
X = np.load(file)
t, *data = X.T

# some constants
tf = t[-1]
N = len(t)
dt = tf/N
estimated_period = 2 * (tau + 1)

# for plot lengths
buffer_periods = 4 # ammount of periods at the end to descart during hilbert transform
target_periods_proportion =  0.1 # proportion of the total ammount of periods to calcualte the hilbert transform over
stationary_estimate_proportion = 0.8 # proportion of periods to count as stationary, estimate

points = int(stationary_estimate_proportion * N) # points in stationary region
h1, h2 = data[0][-points:], data[6][-points:]
h1_phase = np.unwrap(np.angle(hilbert(h1 - h1.mean())))
h2_phase = np.unwrap(np.angle(hilbert(h2 - h2.mean())))

# get integer ammount of periods
total_periods = int(np.floor(h1_phase.max()/(2*np.pi))) # should be close to K
target_periods = int(total_periods * target_periods_proportion)
last_full_period_value = total_periods * np.pi * 2
start_inx = find_point_by_value(h1_phase, last_full_period_value - 2*np.pi*(buffer_periods+target_periods))
end_inx = find_point_by_value(h1_phase, last_full_period_value - 2*np.pi*buffer_periods)
points_per_period = int((end_inx - start_inx)/target_periods)

plot_range = slice(start_inx, end_inx)
h1, h2 = data[0][plot_range], data[6][plot_range]
s1, s2 = data[5][plot_range], data[11][plot_range]

plt.figure(figsize=[10.54,  4.8 ])
plt.subplot(3, 2, 2)
plt.plot(h1)
plt.plot(h2)

plt.subplot(3, 2, 4)
plt.plot(s1)
plt.plot(s2)

plt.subplot(3, 2, (1,5))
plt.plot(h1, s1)
plt.plot(h2, s2)

# find the center
ch1, cs1 = h1.mean(), s1.mean()
ch2, cs2 = h2.mean(), s2.mean()
ch = (ch1+ch2)/2
cs = (cs1+cs2)/2
c = np.array([ch, cs])

plt.plot(ch1, cs1, 'o', color='C0')
plt.plot(ch2, cs2, 'o', color='C1')

ang1 = np.unwrap(np.angle(h1-ch+1j*(s1-cs)))
ang2 = np.unwrap(np.angle(h2-ch + 1j * (s2-cs)))

amp1 = np.sqrt((h1-ch)**2 + (s1-cs)**2)
amp2 = np.sqrt((h2-ch)**2 + (s2-cs)**2)

def sincos(amp, angle):
    return (amp * np.cos(angle), amp * np.sin(angle))

plt.plot(*sincos(amp1, ang1))
plt.plot(*sincos(amp2, ang2))

angles = np.linspace(ang1[0], ang1[-1], 500*target_periods)

amp1_interpolator = interp1d(ang1, amp1, kind='linear', bounds_error=False)
amp1_intp = amp1_interpolator(angles)
amp1_intp = np.ma.array(amp1_intp, mask = np.isnan(amp1_intp))
amp2_interpolator = interp1d(ang2, amp2, kind='linear', bounds_error=False)
amp2_intp = amp2_interpolator(angles)
amp2_intp = np.ma.array(amp2_intp, mask = np.isnan(amp2_intp))

plt.plot(*sincos(amp1_intp, angles), '--')
plt.plot(*sincos(amp2_intp, angles), '--')

amp_diff = amp1_intp-amp2_intp
amp_sum = amp1_intp+amp2_intp
def calc_M(diff, add):
    return np.trapz( amp_diff**2)/np.trapz( amp_sum**2)
M = calc_M(amp_diff, amp_sum)
# M = np.sum( ((amp1-amp2_intp)[:-1]*np.abs(diff(ang1)))**2 )

plt.title(f'{M:.4f}')

# plot rolling M
plt.subplot(3,2,6)
time_M = np.array([calc_M(amp_diff[i:i+points_per_period], amp_sum[i:i+points_per_period]) for i in range((target_periods-1) * 500)])
plt.plot(time_M)

#%% Test with fake signals

def process_M(sx1, sy1, sx2, sy2):
    cx = (sx1.mean() + sx2.mean())/2
    cy = (sy1.mean() + sy2.mean())/2
    
    p1 = np.unwrap(np.angle( sx1 - cx + 1j * (sy1 - cy) ))
    p2 = np.unwrap(np.angle( sx2 - cx + 1j * (sy2 - cy) ))
    
    a1 = np.sqrt( (sx1-cx)**2 + (sy1-cy)**2 )
    a2 = np.sqrt( (sx2-cx)**2 + (sy2-cy)**2 )
    
    a2_interpolator = interp1d(p2, a2, bounds_error=False)
    a2 = a2_interpolator(p1)
    a2 = np.ma.array(a2, mask=np.isnan(a2))
    
    M = np.trapz( (a1-a2)**2, p1) / np.trapz( (a1+a2)**2, p1)
    return M, (cx, cy)

angles = np.linspace(0, 8*2*np.pi, 4000)
fig, axarr = plt.subplots(3, 3, figsize = (10, 10))

# [0,0]
ax = axarr[0,0]
a1 = 1
a2 = 1

ax.plot(*sincos(a1, angles))
ax.plot(*sincos(a2, angles))
M, c = process_M(*sincos(a1, angles), *sincos(a2, angles))
ax.set_title(f'{M:.4f}')
ax.plot(*c, 'x')

# [0,1]
ax = axarr[0,1]
a1 = 0.3
a2 = 3

ax.plot(*sincos(a1, angles))
ax.plot(*sincos(a2, angles))
M, c = process_M(*sincos(a1, angles), *sincos(a2, angles))
ax.set_title(f'{M:.4f}')
ax.plot(*c, 'x')

# [0,2]
ax = axarr[0,2]
a1 = 3
x2, y2 = sincos(a1, angles)
x2 += 1
y2 += 1

ax.plot(*sincos(a1, angles))
ax.plot(x2, y2)
M, c = process_M(*sincos(a1, angles), x2, y2)
ax.set_title(f'{M:.4f}')
ax.plot(*c, 'x')

# [1,0]
ax = axarr[1,0]
a1 = 1
a2 = 0.3*np.sin(angles*2)+1
x2, y2 = sincos(a2, angles)


ax.plot(*sincos(a1, angles))
ax.plot(x2, y2)
M, c = process_M(*sincos(a1, angles), x2, y2)
ax.set_title(f'{M:.4f}')
ax.plot(*c, 'x')

# [1,1]
ax = axarr[1,1]
a1 = 0.3*np.cos(angles*2)+1
a2 = 0.3*np.cos(angles*2-np.pi)+1

ax.plot(*sincos(a1, angles))
ax.plot(*sincos(a2, angles))
M, c = process_M(*sincos(a1, angles), *sincos(a2, angles))
ax.set_title(f'{M:.4f}')
ax.plot(*c, 'x')

# [1,2]
ax = axarr[1,2]
a1 = 1
k = 0.2
a2 = (1-k) + angles * (2*k)/angles.max()

####
da = (a2-a1)**2
sa = (a2+a1)**2
dp = angles[1]-angles[0]

ppp = int(angles.size/8) #pointes per period
moving_M = np.array([np.trapz(da[i:i+ppp], dx=dp)/np.trapz(sa[i:i+ppp], dx=dp) for i in range(ppp*7)])
####

ax.plot(*sincos(a1, angles))
ax.plot(*sincos(a2, angles))
M, c = process_M(*sincos(a1, angles), *sincos(a2, angles))
ax.set_title(f'{M:.4f}')
ax.plot(*c, 'x')

# [2,0]
ax = axarr[2,0]
a1 = 0.5
x2, y2 = sincos(a1, angles)
x2 += 1
y2 += 1

ax.plot(*sincos(a1, angles))
ax.plot(x2, y2)
M, c = process_M(*sincos(a1, angles), x2, y2)
ax.set_title(f'{M:.4f}')
ax.plot(*c, 'x')

# [2,1]
ax = axarr[2,1]

x1, y1 = sincos(2.5, angles)
x1 += 2.5
a2 = 3 * np.cos(angles) + 1 * np.cos(angles*2) + 1

ax.plot(x1, y1)
ax.plot(*sincos(a2, angles))
M, c = process_M(x1, y1, *sincos(a2, angles))
ax.set_title(f'{M:.4f}')
ax.plot(*c, 'x')


# [2,2]
ax = axarr[2,2]
a1 = 1
x2 = np.linspace(-1, 1)
lin = 3*np.abs(x2)-0.5
cuad = 3.5*x2**2-1
x2 = np.concatenate((x2, x2[::-1]))
y2 = np.concatenate((lin, cuad[::-1])) /2 - 0.5

ax.plot(*sincos(a1, angles))
ax.plot(x2, y2)
M, c = process_M(*sincos(a1, angles), x2, y2)
ax.set_title(f'{M:.4f}')
ax.plot(*c, 'x')


plt.tight_layout()

plt.figure()
plt.plot(angles[:moving_M.size], moving_M)