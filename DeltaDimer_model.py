# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:32:07 2020

@author: Marcos
"""

from ddeint import ddeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import interp1d

from utils import new_name, iter_to_csv
    
def model_full(X, t, tau, 
               d, b_H, b_C, b_D, b_N, 
               h, S0, H0, a_H, 
               kCDm, kCDp, kCN, kDN, kEN):
    
    H1, C1, D1, E1, N1, S1, H2, C2, D2, E2, N2, S2 = X(t)
    H1d, _, _, _, _, S1d, H2d, _, _, _, _, S2d = X(t-tau)
    
    model = [
        -d * H1 +  (b_H+ a_H * (S1d/S0)**h )/ ( (1+(S1d/S0)**h) * (1+(H1d/H0)**h)),
        -d * C1 +  b_C / (1+(H1d/H0)**h) + kCDm * E1 - kCDp * C1 * D1 - kCN * C1 * N2,
        -d * D1 +  b_D + kCDm * E1 - kCDp * C1 * D1 - kDN * D1 * N2,
        -d * E1 - kCDm * E1 + kCDp * C1 * D1 - kEN * E1 * N2,
        -d * N1 + b_N - N1 * (kCN * C2 + kDN * D2 + kEN * E2),
        -d * S1 +  N1 * (kCN * C2 + kDN * D2 + kEN * E2),
        
        -d * H2 +  (b_H+ a_H * (S2d/S0)**h )/ ( (1+(S2d/S0)**h) * (1+(H2d/H0)**h)),
        -d * C2 +  b_C / (1+(H2d/H0)**h) + kCDm * E2 - kCDp * C2 * D2 - kCN * C2 * N1,
        -d * D2 +  b_D + kCDm * E2 - kCDp * C2 * D2 - kDN * D2 * N1,
        -d * E2 - kCDm * E2 + kCDp * C2 * D2 - kEN * E2 * N1,
        -d * N2 + b_N - N2 * (kCN * C1 + kDN * D1 + kEN * E1),
        -d * S2 +  N2 * (kCN * C1 + kDN * D1 + kEN * E1),
        ]
    return np.array(model)

def model_adim(X, t, tau, 
               b_H, b_C, b_D, b_N, # b == beta
               eta, a, s, # a == alpha, s == sigma
               lm, lp, kC, kD, kE, #l == lambda, k == kappa, lm == lm/H0
               det, fc): #det == detuning , fc*tau == tauc
    
    h1, c1, d1, e1, n1, s1, h2, c2, d2, e2, n2, s2 = X(t) # unpack variables
    h1d, _, _, _, _, s1d, _, _, _, _, _, _ = X(t-tau) # tau delayed variables
    _, _, _, _, _, _, h2d, _, _, _, _, s2d = X(t-tau*det) # detuning
    h1c, _, _, _, _, _, h2c, _, _, _, _, _ = X(t-tau*fc) # tau c delay
    
    model = [
        -h1 + b_H * (1 + a * (s * s1d)**eta ) / ( (1 + (s * s1d)**eta) * (1 + h1d**eta) ),
        -c1 + b_C / (1 + h1c**eta) + lm * e1 - lp * c1 * d1 - kC * c1 * n2,
        -d1 + b_D + lm * e1 - lp * c1 * d1 - kD * d1 * n2,
        -e1 - lm * e1 + lp * c1 * d1 - kE * e1 * n2,
        -n1 + b_N - n1 * (kC * c2 + kD * d2 + kE * e2),
        -s1 +  n1 * (kC * c2 + kD * d2 + kE * e2),
        
        -h2 + b_H * (1 + a * (s * s2d)**eta ) / ( (1 + (s * s2d)**eta) * (1 + h2d**eta) ),
        -c2 + b_C / (1 + h2c**eta) + lm * e2 - lp * c2 * d2 - kC * c2 * n1,
        -d2 + b_D + lm * e2 - lp * c2 * d2 - kD * d2 * n1,
        -e2 - lm * e2 + lp * c2 * d2 - kE * e2 * n1,
        -n2 + b_N - n2 * (kC * c1 + kD * d1 + kE * e1),
        -s2 +  n2 * (kC * c1 + kD * d1 + kE * e1),
        ]
    return np.array(model)

#### Parameters
CI = [4, 5]

d = 0.3 # 1/min
b_H, b_C, b_D, b_N = 1, 2, 3, 5
kCDp, kCDm = 0, 0
kCN, kDN, kEN = 0.003477, 0.0001, 0.0
H0 = 1 #!!!

parameters = {
    'tau' : 3        ,
    'b_H' : b_H/d/H0  ,
    'b_C' : b_C/d/H0  ,
    'b_D' : b_D/d/H0  ,
    'b_N' : b_N/d/H0  ,
    'eta' : 2.3         , # hill exponent
    'a'   : 5         , # alpha
    's'   : 0.1      , # sigma
    'lm'  : kCDm/d    , # lm == lm/H0
    'lp'  : kCDp*H0/d ,
    'kC'  : kCN*H0/d  ,
    'kD'  : kDN*H0/d  ,
    'kE'  : kEN*H0/d  ,
    'det' : 1.0       ,
    'fc'  : 0      ,
    }

names = ['H1', 'C1', 'D1', 'E1', 'N1', 'S1', 
         'H2', 'C2', 'D2', 'E2', 'N2', 'S2']

def find_point_by_value(array, value):
    return np.abs(array-value).argmin()

def past_values(t):
    return np.concatenate( ( CI[0]*np.ones(6), CI[1]*np.ones(6) ) )

estimated_period = 2 * (parameters['tau'] + 1) # tau is automatically adimentionalized
n = 800 # ammount of points per estimated period
K = 80 # ammount of estiamted periods to integrate over

buffer_periods = 4 # ammount of periods at the end to descart during hilbert transform
target_periods_proportion =  0.3 # proportion of the total ammount of periods to calcualte the hilbert transform over
stationary_estimate_proportion = 0.8 # proportion of periods to count as stationary, estimate

tf = K * estimated_period
N = n * K
dt = tf/N
plot_range = slice(int(-15*estimated_period/dt), N, 2)

times = np.linspace(0, tf, N)
tt = times[plot_range]

Xint = ddeint(model_adim, past_values, times, fargs=parameters.values())

#### Phase stuff

# hilbert transform to get phases
points = int(stationary_estimate_proportion * N) # points in stationary region
h1, h2 = Xint[:,0][-points:], Xint[:, 6][-points:]
s1, s2 = Xint[:,5][-points:], Xint[:,11][-points:]
d1, d2 = Xint[:,2][-points:], Xint[:,8][-points:]
h1_phase = np.unwrap(np.angle(hilbert(h1 - h1.mean())))
h2_phase = np.unwrap(np.angle(hilbert(h2 - h2.mean())))

# get integer ammount of periods for mean and trend scan
h1_phase = np.unwrap(np.angle(hilbert(h1 - h1.mean())))
h2_phase = np.unwrap(np.angle(hilbert(h2 - h2.mean())))

# get integer ammount of periods
total_periods = int(np.floor(h1_phase.max()/(2*np.pi))) # should be close to K*stationary_estimate_proportion
target_periods = int(total_periods * target_periods_proportion)
last_full_period_value = total_periods * np.pi * 2
start_inx = find_point_by_value(h1_phase, last_full_period_value - 2*np.pi*(buffer_periods+target_periods))
end_inx = find_point_by_value(h1_phase, last_full_period_value - 2*np.pi*buffer_periods)
points_per_period = int((end_inx - start_inx)/target_periods)

mean_K = np.mean(K := np.cos(h1_phase[start_inx:end_inx] - h2_phase[start_inx:end_inx]))
moving_mean = np.array([K[i : points_per_period + i].mean() for i in range((target_periods-1) * points_per_period)])

L = moving_mean.size
polytimes = np.linspace(0, L * tf/N, L)
slopeK, _ = np.polyfit(polytimes, moving_mean, 1)


#### Calculating orbit difference
# find the center
ch1, cs1 = h1.mean(), s1.mean()
ch2, cs2 = h2.mean(), s2.mean()
ch = (ch1+ch2)/2
cs = (cs1+cs2)/2
c = np.array([ch, cs])

ang1 = np.unwrap(np.angle(h1-ch+1j*(s1-cs)))
ang2 = np.unwrap(np.angle(h2-ch + 1j * (s2-cs)))

amp1 = np.sqrt((h1-ch)**2 + (s1-cs)**2)
amp2 = np.sqrt((h2-ch)**2 + (s2-cs)**2)

angles = np.linspace(ang1[0], ang1[-1], 500*target_periods)

amp1_interpolator = interp1d(ang1, amp1, kind='linear', bounds_error=False)
amp1_intp = amp1_interpolator(angles)
amp1_intp = np.ma.array(amp1_intp, mask = np.isnan(amp1_intp))
amp2_interpolator = interp1d(ang2, amp2, kind='linear', bounds_error=False)
amp2_intp = amp2_interpolator(angles)
amp2_intp = np.ma.array(amp2_intp, mask = np.isnan(amp2_intp))

amp_diff = amp1_intp-amp2_intp
amp_sum = amp1_intp+amp2_intp
def calc_M(diff, add):
    return np.trapz( amp_diff**2, dx=angles[0]-angles[1])/np.trapz( amp_sum**2, dx=angles[0]-angles[1])
M = calc_M(amp_diff, amp_sum)

time_M = np.array([calc_M(amp_diff[i:i+points_per_period], amp_sum[i:i+points_per_period]) for i in range((target_periods-1) * 500)])
slopeM, _ = np.polyfit(angles[:time_M.size], time_M, 1)


# print('\a')

#### Plot stuff
f = plt.figure(figsize=(16,9))
gs = f.add_gridspec(3, 3)
axarr = [f.add_subplot(gs[i,j]) for i in range(2) for j in range(3)]

sdet = f"detuning={parameters['det']}"
sfc = r'$\tau_c$={:.2f}'.format(parameters['fc'])
f.suptitle(n := f"{kCN=}, {kDN=}, {kEN=}, a={parameters['a']}, tau={parameters['tau']}, s={parameters['s']}\n {sdet}, {sfc}, {CI=}")

for i, ax in enumerate(axarr):
    name1, name2 = names[i], names[i+6]
    x1, x2 = Xint[:,i], Xint[:, i+6]
    ax.plot(tt, x1[plot_range], lw=1.5, label = name1 )
    ax.plot(tt, x2[plot_range], '--' , label = name2 )
    
    ax.set_title(name1[0])
    ax.legend(loc='center right')

ax_long = f.add_subplot(gs[2, :])
ax_long.plot(times, Xint[:,0],lw=1.5)
ax_long.plot(times, Xint[:,6], '--', lw=1.5)
ax_long.set_title('H, full time series')

plt.tight_layout()

f2 = plt.figure(figsize=(10, 9))
gs2 = f2.add_gridspec(3, 2, height_ratios=[1, 1, 2])

ax_K = f2.add_subplot(gs2[0, :])
ax_K.set_title(r'$\langle M \rangle$ = {:.3f}, slope = {:.1e}'.format(mean_K, slopeK))
ax_K.set_xlabel('time [periods]')
ax_K.set_ylabel(r'$cos(\Delta \phi)$')

# redefine time scale
t = times[-points:]
time_per_period = (t[-1]-t[0])/total_periods
t = t[start_inx:end_inx]/time_per_period

ax_K.plot(t, K)
ax_K.plot(t[:moving_mean.size], moving_mean)

ax_M = f2.add_subplot(gs2[1, :])
ax_M.set_title(r'$\langle M \rangle$ = {:.3f}, slope = {:.1e}'.format(M, slopeM))
ax_M.set_xlabel(r'Angle $[2\pi rad]$')
ax_M.set_ylabel(r'$\int \Delta A $ / $ \langle {A} \rangle$')
ax_M.plot(angles[:time_M.size]/(2*np.pi), time_M)


ax_O1 = f2.add_subplot(gs2[2, 0])
ax_O1.set_title('H-S orbit')
ax_O1.set_xlabel('H')
ax_O1.set_ylabel('S')
ax_O1.plot(h1, s1)
ax_O1.plot(h2, s2)

ax_O2 = f2.add_subplot(gs2[2, 1])
ax_O2.set_title('H-D orbit')
ax_O2.set_xlabel('H')
ax_O2.set_ylabel('D')
ax_O2.plot(h1, d1)
ax_O2.plot(h2, d2)

plt.tight_layout()

#### Save stuff
import os

BASE_DIR = 'DeltaDimer_data/tauc_alpha/CI_scan/'

file_name = n.replace('\n', ',').replace('\\', '').replace('$','').replace(',', ';')
file_name = new_name(os.path.join(BASE_DIR, file_name+'.npy'), newformater=' (%d)')
file_name, _ = os.path.splitext(os.path.basename(file_name))

f.savefig(os.path.join(BASE_DIR, 'imgs', file_name+'.png'))
f2.savefig(os.path.join(BASE_DIR, 'imgs', file_name+'_order.png'))

Xsave = np.concatenate((np.expand_dims(times, 1), Xint), axis=1)
np.save(os.path.join(BASE_DIR, file_name+'.npy'), Xsave)

with open(os.path.join(BASE_DIR, 'runs.csv'), 'a') as runs_file:
    runs_file.write(file_name + 
                   ',' + 
                   iter_to_csv(parameters.values(), fmt='.6f') + 
                   f',{CI[0]},{CI[1]}' +
                   f',{mean_K},{slopeK},{M},{slopeM}' +
                   '\n'
                   )
