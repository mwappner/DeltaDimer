# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:29:42 2021

@author: Marcos
"""
import numpy as np
from scipy.signal import hilbert
from scipy import fft
import matplotlib.pyplot as plt
from utils import contenidos, make_dirs_noreplace
import os

BASE_DIR = 'DeltaDimer_data/tauc_alpha/manual_runs/'
files = contenidos(BASE_DIR, filter_ext='.npy', sort='age')

def find_point_by_value(array, value):
    return np.abs(array-value).argmin()

#%% Scans

current_dir = 6
BASE_DIR = 'DeltaDimer_data/tauc_alpha/'
BASE_DIR = os.path.join(BASE_DIR, str(current_dir))
make_dirs_noreplace(os.path.join(BASE_DIR, 'synch_imgs'))

files = contenidos(BASE_DIR, filter_ext='.npy')

#%% One signal

inx = 0
file = files[inx]

X = np.load(file)
t, *data = X.T

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

s = slice(-10000,None)
ax1.set_title('Time series', fontsize=15)
ax1.plot(tt := t[s], pd1 := data[0][s])
ax1.set_ylabel('Amp', fontsize=15)

zeroed_data = pd1 - pd1.mean()
analytic1 = hilbert(zeroed_data)

lin1 = np.polyfit(tt, p1 := np.unwrap(np.angle(analytic1)), 1)

ax2.set_title(r'Instant phase $\phi(t)$', fontsize=15)
ax2.plot(tt, np.polyval(lin1, tt), 'k--', lw=2)
ax2.plot(tt, p1, lw=2)
ax2.set_ylabel('[rad]', fontsize=15)

# detrended
ax3.plot(tt, p1-np.polyval(lin1, tt), lw=2)

ax3.set_title(r'$\phi (t) - \omega t$', fontsize=15)
ax3.set_xlabel('Time [d^-1]', fontsize=15)
ax3.set_ylabel('[rad]', fontsize=15)

ax3.set_xlim(tt[0]*1.1, tt[-1]*0.8)
plt.tight_layout()


#%% Two signals, single plot
inx = -11
file = files[inx]

X = np.load(file)
t, *data = X.T

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

s = slice(-10000,None)
ax1.set_title('Time series')
ax1.plot(tt := t[s], pd1 := data[0][s])
ax1.plot(tt, pd2 := data[6][s])

zeroed_data = pd1 - pd1.mean()
analytic1 = hilbert(zeroed_data)

zeroed_data = pd2 - pd2.mean()
analytic2 = hilbert(zeroed_data)

# plt.plot(analytic.real)
# plt.plot(analytic.imag)
# ax1.plot(np.abs(analytic1)+pd1.mean())
# ax1.plot(np.abs(analytic2)+pd2.mean())

ax2.set_title('Instant phase')
ax2.plot(tt, p1 := np.unwrap(np.angle(analytic1)))
ax2.plot(tt, p2 := np.unwrap(np.angle(analytic2)))

ax3.set_title('Phase difference')
ax3.set_xlabel('Time [d^-1]')
ax3.plot(tt, np.cos(p1-p2), 'k')

ax3.set_xlim(tt[0]*1.1, tt[-1]*0.9)
plt.tight_layout()

# detrended
plt.figure()
lin1 = np.polyfit(tt, p1, 1)
lin2 = np.polyfit(tt, p2, 1)

plt.plot(tt, p1-np.polyval(lin1, tt))
plt.plot(tt, p2-np.polyval(lin2, tt))

print(np.abs(np.mean(np.cos(p1-p2))))
#%%

cos = np.cos(np.linspace(0, 30, 200))
plt.plot(cos)
plt.plot(np.unwrap(np.arccos(cos)))

#%% making all plots

for file in files:
    X = np.load(file)
    t, *data = X.T
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        
    s = slice(-10000,None)
    ax1.set_title('Time series')
    ax1.plot(tt := t[s], pd1 := data[0][s])
    ax1.plot(tt, pd2 := data[6][s])
    
    zeroed_data = pd1 - pd1.mean()
    analytic1 = hilbert(zeroed_data)
    
    zeroed_data = pd2 - pd2.mean()
    analytic2 = hilbert(zeroed_data)
    
    ax2.set_title('Instant phase')
    ax2.plot(tt, p1 := np.unwrap(np.angle(analytic1)))
    ax2.plot(tt, p2 := np.unwrap(np.angle(analytic2)))
    
    ax3.set_title('Phase difference')
    ax3.set_xlabel('Time [d^-1]')
    ax3.plot(tt, np.cos(p1-p2), 'k')
    
    ax3.set_xlim(tt[0]*1.1, tt[-1]*0.9)
    
    threshold = 0.99
    mean_cos_phase_diff = np.mean(np.cos(p1-p2))
    if mean_cos_phase_diff>threshold :
        synch_type = 'in phase'
    elif mean_cos_phase_diff<-threshold:
        synch_type = 'in antiphase'
    else:
        synch_type = 'none'
        
    fig.suptitle(f'Mean cosine phase dif = {mean_cos_phase_diff:.4f}\nSynchronization type: {synch_type}')
    plt.tight_layout()
    
    plt.savefig(os.path.join(BASE_DIR, 'imgs_synch', os.path.basename(file)[:-3]+'jpg'))
    plt.close(fig)
    
#%% trends

file = 'kCN=0.001; kDN=0.0; kEN=0.0; a=5; tau=3; s=0.01; detuning=1.0; tau_c=tau_0.00; CI=[1; 5].npy'
file = os.path.join(BASE_DIR, file)

target_periods = 5
buffer_periods = 3
total_periods_to_scan = 70

X = np.load(file)
t, *data = X.T
tt = t[:]

# estimated_period = 2 * (3+1) # 2*(tau+1)
# points_in_a_period = estimated_period / t[1]

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

ax1.plot(tt, pd1 := data[0][:])
ax1.plot(tt, pd2 := data[6][:])
ax1.set_title('Time series')

zeroed_data = pd1 - pd1.mean()
analytic1 = hilbert(zeroed_data)

zeroed_data = pd2 - pd2.mean()
analytic2 = hilbert(zeroed_data)

ax2.plot(tt, p1 := np.unwrap(np.angle(analytic1)))
ax2.plot(tt, p2 := np.unwrap(np.angle(analytic2)))
ax2.set_title('Instant phase')

ax3.plot(tt, phase_dif := np.cos(p1-p2), 'k')
ax3.set_xlabel('Time')
ax3.set_title('Cosine phase difference')

total_periods = np.floor(p1.max()/(2*np.pi))
last_full_period_value = total_periods * np.pi * 2
start_inx = find_point_by_value(p1, last_full_period_value - 2*np.pi*(buffer_periods+target_periods))
end_inx = find_point_by_value(p1, last_full_period_value - 2*np.pi*buffer_periods)
points_per_period = int(( end_inx - start_inx ) / target_periods)
first_point = int(- points_per_period * (total_periods_to_scan + buffer_periods))

moving_mean = np.array([phase_dif[first_point + i : first_point + 2*points_per_period + i].mean() for i in range(total_periods_to_scan * points_per_period)])

plt.plot(tt[first_point : first_point + moving_mean.size], moving_mean)

plt.tight_layout()
# plt.plot([tt[first_point], tt[first_point + moving_mean.size]], [moving_mean[0], moving_mean[-1]])

### Figure out frequency with fft
# N_x = len(phase_dif)
# tf_x = t[-1] - t[points]

# # real part of FFT without frequency 0
# fx = 2/N_x * np.abs(fft.fft(phase_dif))[1:N_x//2]
# ffreq = fft.fftfreq(N_x, tf_x/N_x)[1:N_x//2]
# plt.figure()
# plt.plot(ffreq, fx)

#%% fixed ammount of periods

target_periods = 5
buffer_periods = 2

inx = -11
file = files[inx]

X = np.load(file)
t, *data = X.T

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(pd1 := data[0][:])
ax1.plot(pd2 := data[6][:])

zeroed_data = pd1 - pd1.mean()
analytic1 = hilbert(zeroed_data)

zeroed_data = pd2 - pd2.mean()
analytic2 = hilbert(zeroed_data)

ax2.plot(p1 := np.unwrap(np.angle(analytic1)))
ax2.plot(p2 := np.unwrap(np.angle(analytic2)))

ax3.plot(pd := p1-p2)

total_periods = np.floor(p1.max()/(2*np.pi))
last_full_period_value = total_periods * np.pi * 2
start_inx = find_point_by_value(p1, last_full_period_value - 2*np.pi*(buffer_periods+target_periods))
end_inx = find_point_by_value(p1, last_full_period_value - 2*np.pi*buffer_periods)

ax1.axvline(start_inx, color='k')
ax1.axvline(end_inx, color='k')

print(np.abs(np.mean(pd[start_inx:end_inx])))



