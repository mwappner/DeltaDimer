# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:06:33 2020

@author: Marcos
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib.colors import ListedColormap, TwoSlopeNorm

import numpy as np
import pandas as pd
import functools 
from scipy import interpolate

def big_or(column, values):
    return functools.reduce(lambda x, y: x|y, (column==v for v in values))

def filter_nans(df, column=None):
    """ Takes in a dataframe and returns a new one where all rows with nans have 
    been removed. If column is given, the filering is only one in said column
    """
    if column:
        return df[np.logical_not(np.isnan(df[column]))]
    else:
        bools = [np.logical_not(np.isnan(df[column])) for column in df.columns]
        where = functools.reduce(lambda x, y: x&y, bools)
    
        return df[where]        
    

def filter_data(df, filters):
    """ Takes in a dictionary with keys being the parameter to check and values 
    being the values to match for that parameter, either floats or tuples, as 
    well as the dataframe to filter. Returns the filtered dataframe."""

    # make floats into iterables
    filters = {k: (v if hasattr(v, '__iter__') else [v]) for k, v in filters.items()}
    
    # create bool masks
    bools = [big_or(df[k], v) for k, v in filters.items()]
    
    # make a big and
    where = functools.reduce(lambda x, y: x&y, bools)
    
    return df[where]

def filter_single(df, d, b, h, tau, x0=1, plot=False):
    
    this_filter = {'d':d, 'b':b, 'h':h, 'tau':tau, 'x0':x0}
    data = filter_data(df, this_filter )
    
    if plot:
        I = plt.imread('regulacion_imgs/{}.png'.format(', '.join([f'{k}={v}' for k, v in this_filter.items()])))
        plt.imshow(I)
        plt.axis('off')
        plt.title('amp = {:.2e}'.format(data['amp'].values[0]))
    
    return data

# %% Load data, sort by 'd' and store parameter values

data = pd.read_csv('regulacion_imgs/regulacion.csv').sort_values(by=['d'])

parameters = data.keys()[:5]
parameter_values = {k:sorted(list(set(data[k]))) for k in parameters}


#%% plot for a few parameter values

# d = [0.5, 1, 1.3]
b = 20
h = 3
tau = parameter_values['tau']
# tau = [1, 5, 10, 17]
# tau = [1,2,3, 4, 5, 6]

for t in tau:
    pp = filter_data(data, {'tau':t, 'b':b, 'h':h})
    plt.plot(pp['d'], pp['amp'], 'o-', label = r'$\tau$ = {}'.format(t))
plt.legend()
plt.ylim([0, 100])


#%% Scatter

b = 1
h = 3

# scatter plot
plt.figure()
section = filter_data(data, {'b':b, 'h':h})
plt.scatter(x=data['d'], y=data['tau'], c=data['amp'], vmin=0, vmax=12)

# interpolated
plt.figure()
data_nanless = filter_data(data, {'b':b, 'h':h})
# data_nanless = filter_nans(filter_data(data, {'b':b, 'h':h}), column='amp')


dmin = min(parameter_values['d'])
dmax = max(parameter_values['d'])
tmin = min(parameter_values['tau'])
tmax = max(parameter_values['tau'])

xgrid, ygrid = np.mgrid[dmin:dmax:0.01, tmin:tmax:1]
interpolated = interpolate.griddata(
                            (data_nanless['d'], data_nanless['tau']), 
                            data_nanless['amp'], 
                            (xgrid, ygrid),
                            method='nearest'
                                    )
plt.imshow(interpolated.T, extent=(dmin, dmax, tmin, tmax), aspect='auto',vmin=0, vmax=12, origin='lower')
plt.colorbar()

# 3D scatter
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.scatter(xs=data['d'], ys=data['tau'], zs=data['amp'], c=data['amp'], vmin=0, vmax=12)
ax.set_zlim(0, 10)
ax.set_xlabel('d')
ax.set_ylabel(r'$\tau$')
ax.set_zlabel('amp')

#%% Plot all b and h


dmin = min(parameter_values['d'])
dmax = max(parameter_values['d'])
tmin = min(parameter_values['tau'])
tmax = max(parameter_values['tau'])

xgrid, ygrid = np.mgrid[dmin:dmax:0.01, tmin:tmax:1]

fig, axarray = plt.subplots(2, 3, figsize=(18,9))
fig.suptitle('Global color scale')

fig2, axarray2 = plt.subplots(2, 3, figsize=(18,9))
fig2.suptitle('Individual color scales')

MAX = 350
# max_values = [[1, 2, 4], [1, 3, 7], [2, 2, 9], [2, 3, 12], [3, 2, 13], [3, 3, 17]]
max_values = [[20, 2, 100], [20, 3, 130], [35, 2, 160], [35, 3, 220], [60, 2, 250], [60, 3, 340]]
max_values = pd.DataFrame(max_values, columns=['b', 'h', 'max'])

for i, b in enumerate(parameter_values['b']):
    for j, h in enumerate(parameter_values['h']):
        ax = axarray[j, i]
        ax2 = axarray2[j, i]
        
        # data_nanless = filter_data(data, {'b':b, 'h':h})
        data_nanless = filter_nans(filter_data(data, {'b':b, 'h':h, 'x0':1}), column='amp')
        
        interpolated = interpolate.griddata(
                            (data_nanless['d'], data_nanless['tau']), 
                            data_nanless['amp'], 
                            (xgrid, ygrid),
                            method='nearest'
                                    )
        ax.imshow(interpolated.T, 
                  extent=(dmin, dmax, tmin, tmax), 
                  aspect='auto',
                  vmin=0, vmax=MAX, 
                  origin='lower',
                   cmap = cm.hot
                  )
        # plt.colorbar()        
        ax.set_xlabel('d')
        ax.set_ylabel(r'$\tau$')        
        ax.set_title(f'{b = }, {h = }')
        
        max_local_value = filter_data(max_values, {'b':b, 'h':h})['max'].values[0]
        I = ax2.imshow(interpolated.T, 
                  extent=(dmin, dmax, tmin, tmax), 
                  aspect='auto',
                  vmin=0, vmax=max_local_value, 
                  origin='lower',
                   cmap = cm.hot
                   )
        plt.colorbar(I, ax=ax2)
        
        ax2.set_xlabel('d')
        ax2.set_ylabel(r'$\tau$')
        ax2.set_title(f'{b = }, {h = }')


#%% Frequencies

def approx_period(d, tau):
    return 2*tau + 2/d

# d = [0.5, 1, 1.3]
b = 20
h = 3
# tau = parameter_values['tau']
tau = [1, 5, 10, 17]
# tau = [1,2,3, 4, 5, 6]

fig, (ax1, ax2) = plt.subplots(1, 2)

for t in tau:
    pp = filter_data(data, {'tau':t, 'b':b, 'h':h})
    pp_osc = pp[pp['amp']>0.1]
    ax1.plot(pp_osc['d'], pp_osc['period'], 'o-', label = r'$\tau$ = {}'.format(t))
    
    
    ax2.plot(pp['d'], approx_period(pp['d'], t) )
    
# ax1.set_ylim([0, 10])
ax1.set_ylabel('Frequencies')
ax1.set_xlabel('d')
ax1.legend()

ax2.set_ylim(ax1.get_ylim())

#%% d vs tau image

MIN_AMP = 10
MAX_AMP = 500

b = 20
h = 3

def approx_period(d, tau):
    return 2*tau + 2/d

dmin = min(parameter_values['d'])
dmax = max(parameter_values['d'])
tmin = min(parameter_values['tau'])
tmax = max(parameter_values['tau'])

dgrid, taugrid = np.mgrid[dmin:dmax:0.01, tmin:tmax:1]

xgrid, ygrid = np.mgrid[dmin:dmax:0.01, tmin:tmax:1]

period_theoretical = approx_period(dgrid, taugrid)
      
# data_nanless = filter_data(data, {'b':b, 'h':h})
data_nanless = filter_nans(filter_data(data, {'b':b, 'h':h}), column='amp')

amplitudes = interpolate.griddata(
                    (data_nanless['d'], data_nanless['tau']), 
                    data_nanless['amp'], 
                    (xgrid, ygrid),
                    method='nearest'
                            )

mask = np.logical_and(amplitudes>MIN_AMP, amplitudes<MAX_AMP)

period_meassured = interpolate.griddata(
                    (data_nanless['d'], data_nanless['tau']), 
                    data_nanless['period'], 
                    (xgrid, ygrid),
                    method='nearest'
                            )

period_meassured[np.logical_not(mask)] = np.nan
period_theoretical[np.logical_not(mask)] = np.nan

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

I1 = ax1.imshow(period_theoretical.T, 
          extent=(dmin, dmax, tmin, tmax), 
          aspect='auto',
           # vmin=0, vmax=50, 
           origin='lower',
           interpolation = 'none'
          # cmap = cm.hot
          )
ax1.set_title('Theoretical')
plt.colorbar(I1, ax=ax1)

I2 = ax2.imshow(period_meassured.T, 
          extent=(dmin, dmax, tmin, tmax), 
          aspect='auto',
           # vmin=0, vmax=50, 
          origin='lower',
          # cmap = cm.hot,
          interpolation = 'none'
          )
ax2.set_title('Meassured')
plt.colorbar(I2, ax=ax2)

difference = 1- period_meassured / period_theoretical

colores = cm.get_cmap('BrBG', 256)(np.linspace(0,1,256))
colores[:, :3] = 1-colores[:, :3]
custom_cm = ListedColormap(colores)

I3 = ax3.imshow(difference.T, 
          extent=(dmin, dmax, tmin, tmax), 
          aspect='auto',
           # vmin=0, vmax=50, 
          origin='lower',
          cmap = custom_cm,
          norm=TwoSlopeNorm(0),
          interpolation = 'none'
          )
ax3.set_title('Percent difference (theoretical-meassured)')
plt.colorbar(I3, ax=ax3)

# %% Frequencies for all b and h

MIN_AMP = 10
MAX_AMP = 500
MAX_PERCENT = 0.3

b = 2
h = 3

def approx_period(d, tau):
    return 2*tau + 2/d

dmin = min(parameter_values['d'])
dmax = max(parameter_values['d'])
tmin = min(parameter_values['tau'])
tmax = max(parameter_values['tau'])

dgrid, taugrid = np.mgrid[dmin:dmax:0.01, tmin:tmax:1]

xgrid, ygrid = np.mgrid[dmin:dmax:0.01, tmin:tmax:1]

period_theoretical = approx_period(dgrid, taugrid)
      
fig_theo, axarray_theo = plt.subplots(2, 3, figsize=(18,9))
fig_theo.suptitle(r'Theoretical period ($2\tau + 2/d$')

fig_exp, axarray_exp = plt.subplots(2, 3, figsize=(18,9))
fig_exp.suptitle('Meassured period')

fig_diff, axarray_diff = plt.subplots(2, 3, figsize=(18,9))
fig_diff.suptitle('Percent difference (theoretical-meassured)')
# fig2, axarray2 = plt.subplots(2, 3, figsize=(18,9))

# fig2.suptitle('Individual color scales')

        
colores = cm.get_cmap('BrBG', 256)(np.linspace(0,1,256))
colores[:, :3] = 1-colores[:, :3]
custom_cm = ListedColormap(colores)

for i, b in enumerate(parameter_values['b']):
    for j, h in enumerate(parameter_values['h']):
        ax_t = axarray_theo[j, i]
        ax_e = axarray_exp[j, i]
        ax_d = axarray_diff[j, i]
        # ax2 = axarray2[j, i]

        # data_nanless = filter_data(data, {'b':b, 'h':h})
        data_nanless = filter_nans(filter_data(data, {'b':b, 'h':h}), column='amp')
        
        amplitudes = interpolate.griddata(
                            (data_nanless['d'], data_nanless['tau']), 
                            data_nanless['amp'], 
                            (xgrid, ygrid),
                            method='nearest'
                                    )
        
        mask = np.logical_and(amplitudes>MIN_AMP, amplitudes<MAX_AMP)
        
        period_meassured = interpolate.griddata(
                            (data_nanless['d'], data_nanless['tau']), 
                            data_nanless['period'], 
                            (xgrid, ygrid),
                            method='nearest'
                                    )
        
        period_theoretical = approx_period(dgrid, taugrid)
        period_meassured[np.logical_not(mask)] = np.nan
        period_theoretical[np.logical_not(mask)] = np.nan

        difference = 1- period_meassured / period_theoretical
        
        I_t = ax_t.imshow(period_theoretical.T, 
          extent=(dmin, dmax, tmin, tmax), 
          aspect='auto',
           # vmin=0, vmax=50, 
           origin='lower',
           interpolation = 'none'
          # cmap = cm.hot
          )
        I_e = ax_e.imshow(period_meassured.T, 
                  extent=(dmin, dmax, tmin, tmax), 
                  aspect='auto',
                   # vmin=0, vmax=50, 
                  origin='lower',
                  # cmap = cm.hot,
                  interpolation = 'none'
                  )
        
        I_d = ax_d.imshow(difference.T, 
                  extent=(dmin, dmax, tmin, tmax), 
                  aspect='auto',
                  origin='lower',
                  cmap = custom_cm,
                    # norm=TwoSlopeNorm(0,vmin=-0.1, vmax=MAX_PERCENT),
                    norm=TwoSlopeNorm(0),
                  interpolation = 'none'
                  )
        
        for ax, I in zip( (ax_t, ax_e, ax_d), (I_t, I_e, I_d)):
            ax.set_xlabel('d')
            ax.set_ylabel(r'$\tau$')        
            ax.set_title(f'{b = }, {h = }')
            plt.colorbar(I, ax=ax)


#%%

############# DO STUFF WITH b d SCAN ################


#%% plot for a few parameter values

# d = [0.5, 1, 1.3]
h = 2
tau = 15 # 6, 10, 15
x0 = 1
# b = [10, 30, 40]
b = parameter_values['b']

for b in b:
    pp = filter_data(data, {'tau':tau, 'b':b, 'h':h, 'x0':x0})
    plt.plot(pp['d'], pp['amp'], 'o-', label = r'b = {}'.format(b))
plt.legend()
# plt.ylim([0, 10])
            
            
#%% Plot all tau and h for x0 = 1 , 2


dmin = min(parameter_values['d'])
dmax = max(parameter_values['d'])
bmin = min(parameter_values['b'])
bmax = max(parameter_values['b'])

xgrid, ygrid = np.mgrid[dmin:dmax:0.01, bmin:bmax:1]

fig, axarray = plt.subplots(2, 3, figsize=(18,9))
fig.suptitle('Global color scale')

fig2, axarray2 = plt.subplots(2, 3, figsize=(18,9))
fig2.suptitle('Individual color scales')

MAX = 250
max_values = [[6, 2, 80], [10, 2, 130], [15, 2, 190], [6, 3, 110], [10, 3, 170], [15, 3, 250]]
max_values = pd.DataFrame(max_values, columns=['tau', 'h', 'max'])

for i, tau in enumerate(parameter_values['tau']):
    for j, h in enumerate(parameter_values['h']):
        ax = axarray[j, i]
        ax2 = axarray2[j, i]
        
        # data_nanless = filter_data(data, {'b':b, 'h':h})
        data_nanless = filter_nans(filter_data(data, {'tau':tau, 'h':h, 'x0':2}), column='amp')
        
        interpolated = interpolate.griddata(
                            (data_nanless['d'], data_nanless['b']), 
                            data_nanless['amp'], 
                            (xgrid, ygrid),
                            method='nearest'
                                    )
        ax.imshow(interpolated.T, 
                  extent=(dmin, dmax, tmin, tmax), 
                  aspect='auto',
                  vmin=0, vmax=MAX, 
                  origin='lower',
                   cmap = cm.hot
                  )
        # plt.colorbar()        
        ax.set_xlabel('d')
        ax.set_ylabel('b')        
        ax.set_title(r'$\tau$ = {}, h = {}'.format(tau, h))
        
        max_local_value = filter_data(max_values, {'tau':tau, 'h':h})['max'].values[0]
        I = ax2.imshow(interpolated.T, 
                  extent=(dmin, dmax, tmin, tmax), 
                  aspect='auto',
                  vmin=0, vmax=max_local_value, 
                  origin='lower',
                    cmap = cm.hot
                    )
        plt.colorbar(I, ax=ax2)
        
        ax2.set_xlabel('d')
        ax2.set_ylabel('b')
        ax2.set_title(r'$\tau$ = {}, h = {}'.format(tau, h))