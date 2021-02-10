# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:53:26 2021

@author: Marcos
"""

import numpy as np
import functools
from itertools import product
import pandas as pd
import os

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.widgets import Slider, TextBox

from utils import contenidos, new_name, find_numbers, Testimado


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
#%% FOR THE NO TAUC SCANS


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
    
#%% FOR THE TAUC-ALPHA SCANS

# BASE_DIR = '/home/user/Documents/Doctorado/DeltaDimer/DeltaDimer_data/tauc_alpha/CI[1,2]/'
# BASE_DIR = '/home/user/Documents/Doctorado/DeltaDimer/DeltaDimer_data/tauc_alpha/CI[1,5]/'
# BASE_DIR = '/home/user/Documents/Doctorado/DeltaDimer/DeltaDimer_data/tauc_alpha/CI[1,2]s/'
# BASE_DIR = '/home/user/Documents/Doctorado/DeltaDimer/DeltaDimer_data/tauc_alpha/CI[1,5]s/'
BASE_DIR = '/home/user/Documents/Doctorado/DeltaDimer/DeltaDimer_data/tauc_alpha/manual_runs/'

files = contenidos(BASE_DIR, filter_ext='.npy', sort='age')
runs = pd.read_csv(BASE_DIR + 'runs.csv', index_col='name')

parameters = runs.keys()
parameter_values = {k:sorted(list(set(runs[k]))) for k in parameters}

#%% Make all images
tau = 3 # HARDCODED!!

NUMBER = 0
X = np.load(files[NUMBER])
t, *data = X.T
name = os.path.basename(files[NUMBER])
# fig_name = name.replace('tau_c', r'$\tau_c$').replace('tau', r'$\tau$').replace('; det', '\ndet')
numbers = find_numbers(name)
fig_name = 'kCN={}, kDN={}, kEN={}, a={}, $\\tau$={}, s={}\ndetuning={}, $\\tau_c$=$\\tau$*{}, CI=[{},{}]'.format(*numbers)

tf = t[-1]
N = len(t)
dt = tf/N
estimated_period = 2 * (tau + 1)
plot_range = slice(int(-10*estimated_period/dt), N)


tt = t[plot_range]

names = ['H1', 'C1', 'D1', 'E1', 'N1', 'S1', 
         'H2', 'C2', 'D2', 'E2', 'N2', 'S2']

f = plt.figure(figsize=(16,9))
gs = f.add_gridspec(3, 3)
axarr = [f.add_subplot(gs[i,j]) for i in range(2) for j in range(3)]

f.suptitle(fig_name)

lines1 = []
lines2 = []
long = []

for i, ax in enumerate(axarr):
    name1, name2 = names[i], names[i+6]
    x1, x2 = data[i], data[i+6]
    lines1.append(ax.plot(tt, x1[plot_range], lw=1.5, label = name1 ))
    lines2.append(ax.plot(tt, x2[plot_range], '--' , lw=1.5, label = name2 ))
    
    ax.set_title(name1[0])
    ax.legend(loc='center right')
    ax.grid()

ax_long = f.add_subplot(gs[2, :])
long.append(ax_long.plot(t, data[0],lw=1.5))
long.append(ax_long.plot(t, data[6], '--', lw=1.5))
ax_long.set_title('H, full time series')
ax_long.grid()

plt.tight_layout()

t_est = Testimado(len(files))
for i, file in enumerate(files):
    
    if i%10==0:
        print(f'Running {i} ot of {len(files)}. ETA = {t_est.time_str(i, "M")} mins')
    
    X = np.load(file)
    t, *data = X.T
    name = os.path.basename(file)
    numbers = find_numbers(name)
    fig_name = 'kCN={}, kDN={}, kEN={}, a={}, $\\tau$={}, s={}\ndetuning={}, $\\tau_c$=$\\tau$*{}, CI=[{},{}]'.format(*numbers)
    
    f.suptitle(fig_name)
    
    for i, ((line1,), (line2,)) in enumerate(zip(lines1, lines2)):
        line1.set_ydata(d:=(data[i][plot_range]))
        m1 = d.min()
        M1 = d.max()
        line2.set_ydata(d:=(data[i+6][plot_range]))
        relim(axarr[i], max(M1, d.max()), min(m1, d.min()), min_size=0.01)
    
    M = []
    m = []
    for k, (line, ) in enumerate(long):
        line.set_ydata(d:=(data[k*6]))
        M.append(d.max())
        m.append(d.min())
    relim(ax_long, max(M), min(m))
    
    
    png_name = new_name(BASE_DIR + 'imgs/' + name[:-4] +'.png')
    plt.savefig(png_name)
    
    
# plt.close(f)

#%% Open one image

# file = BASE_DIR + one_by_params(0.15, 0.1, 0.15, 5, 3, 1.0, 0, [1,5])+'.npy'

some_s = filter_by_values(runs, kCN=0.4, kDN=0.15, kEN=0.1, a=2)

FONT_SIZE = 20
TITLE_SIZE = 25

MAX_IMGS = 10 # maximum ammount of images to open at once, in case I screw up the filter
count = 0

for name, parameters in some_s.iterrows():
    
    tau = parameters['tau']
    
    ci1, ci2 = int(parameters["CI1"]), int(parameters["CI2"])
    file = BASE_DIR + name.replace(f'CI=[{ci1}; {ci2}]', f'CI=[{ci1}, {ci2}]') + '.npy'
    
    X = np.load(file)
    t, *data = X.T
    name = os.path.basename(file)
    # fig_name = name.replace('tau_c', r'$\tau_c$').replace('tau', r'$\tau$').replace('; det', '\ndet')
    numbers = find_numbers(name)
    fig_name = 'kCN={}, kDN={}, kEN={}, a={}, $\\tau$={}\ndetuning={}, $\\tau_c$=$\\tau$*{}, CI=[{},{}]'.format(*numbers)
    
    tf = t[-1]
    N = len(t)
    dt = tf/N
    estimated_period = 2 * (tau + 1)
    plot_range = slice(int(-10*estimated_period/dt), N)
    
    tt = t[plot_range]
    
    names = ['H1', 'C1', 'D1', 'E1', 'N1', 'S1', 
             'H2', 'C2', 'D2', 'E2', 'N2', 'S2']
    
    f = plt.figure(figsize=(16,9))
    gs = f.add_gridspec(3, 3)
    axarr = [f.add_subplot(gs[i,j]) for i in range(2) for j in range(3)]
    
    f.suptitle(fig_name, fontsize=TITLE_SIZE)
    
    lines1 = []
    lines2 = []
    long = []
    
    for i, ax in enumerate(axarr):
        name1, name2 = names[i], names[i+6]
        x1, x2 = data[i], data[i+6]
        lines1.append(ax.plot(tt, x1[plot_range], lw=1.5, label = name1 ))
        lines2.append(ax.plot(tt, x2[plot_range], '--' , lw=1.5, label = name2 ))
        
        ax.set_title(name1[0], fontsize=FONT_SIZE)
        ax.legend(loc='center right')
        ax.tick_params(labelsize=FONT_SIZE*0.6)
    
    ax_long = f.add_subplot(gs[2, :])
    long.append(ax_long.plot(t, data[0],lw=1.5))
    long.append(ax_long.plot(t, data[6], '--', lw=1.5))
    ax_long.set_title('H, full time series', fontsize=FONT_SIZE)
    ax_long.set_ylabel('Amplitude [$H_0$]', fontsize=FONT_SIZE*0.8)
    ax_long.set_xlabel('Time [$d^{-1}$]', fontsize=FONT_SIZE*0.8)
    
    plt.tight_layout()
    
    count += 1
    if count == 10:
        break
    
#%% FOR REGULATORY FUNCTION


def reg_func(s, h, sigma, alpha, eta):
    return (1 + alpha * sigma**eta * s**eta) / ( (1 + sigma**eta * s**eta) * (1 + h**eta) )

eta = 2.3
alpha = 5
sigma = 1

H = np.arange(0, 5, 0.01)
S = np.arange(0, 25, 0.01)
H, S = np.meshgrid(H, S)
r = reg_func(S, H, sigma, alpha, eta)

## Initializing the plot
fig = plt.figure()
plt.get_current_fig_manager().window.setGeometry(*(496, 136, 952, 873)) #sets position and size
ax = fig.add_subplot(projection='3d')

surf = [ax.plot_surface(H, S, r, cmap='viridis', edgecolor='none', alpha=0.8)]
ax.set_title('Regulatory function', fontsize=15)
ax.set_xlabel('h')
ax.set_ylabel('s')

# Move the plot a bit to the right and set view angle
plt.subplots_adjust(left=0.2)
ax.view_init(19.9, -43.2)

## Making the sliders

# eta slider
axeta = plt.axes([0.05, 0.25, 0.0225, 0.63])
eta_slider = Slider(
    ax=axeta,
    label=r'$\eta$',
    valmin=1,
    valmax=20,
    valinit=eta,
    orientation='vertical'
)

# alpha slider
axalpha = plt.axes([0.1, 0.25, 0.0225, 0.63])
alpha_slider = Slider(
    ax=axalpha,
    label=r'$\alpha$',
    valmin=1,
    valmax=10,
    valinit=alpha,
    orientation='vertical'
)

# sigma slider
axsigma = plt.axes([0.15, 0.25, 0.0225, 0.63])
sigma_slider = Slider(
    ax=axsigma,
    label=r'log($\sigma$)',
    valmin=-4,
    valmax=4,
    valinit=np.log10(sigma),
    orientation='vertical'
)

def updater(val):
    surf[0].remove()
    surf[0] = ax.plot_surface(H, S, reg_func(S, H, 10**sigma_slider.val, alpha_slider.val, eta_slider.val), cmap='viridis', edgecolor='none', alpha=0.8)

eta_slider.on_changed(updater)
alpha_slider.on_changed(updater)
sigma_slider.on_changed(updater)

#%% Plot trajectories in 3D

def reg_func(s, h, sigma, alpha, eta):
    return (1 + alpha * sigma * s**eta) / ( (1 + sigma * s**eta) * (1 + h**eta) )

def s_reg_func(s, sigma, alpha, eta):
    return (1 + alpha * sigma * s**eta) / (1 + sigma * s**eta)

runs = pd.read_csv(BASE_DIR + 'runs.csv', index_col='name')
files.update()

# =============================================================================
# ### For the older runs
# # some_s = filter_by_values(runs, kCN=0.4, kDN=0.15, kEN=0.1, a=2, s=0.1)
# 
# # INX = 0 # what index out of the ones in some_s to check
# # #get parameters
# # eta = some_s['eta'].values[INX]
# # alpha = some_s['a'].values[INX]
# # sigma = some_s['s'].values[INX]
# # tau = some_s['tau'].values[INX]
# # name = some_s.index.values[INX]
# =============================================================================
# =============================================================================
# ### For manual runs
FILE_INDX = 6 # newest one
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

# fetch plots
_, ax_im = plt.subplots(figsize=(16,9))
I = plt.imread(BASE_DIR + 'imgs/' + name + '.png')
ax_im.imshow(I)
ax_im.axis('off')
plt.tight_layout()

# load and unpack data
X = np.load(file)
t, *data = X.T

# get data limits to plot and generate regulatory function
h_max = max(data[0].max(), data[6].max())
s_max = max(data[5].max(), data[11].max())

## Initializing the plot
fig = plt.figure(constrained_layout=True)
plt.get_current_fig_manager().window.setGeometry(*(200, 130, 1400, 850)) #sets position and size
fig.suptitle(fig_name, fontsize=15)

gs = fig.add_gridspec(3, 5)

ax3D = fig.add_subplot(gs[:, :3], projection='3d')
ax3D.set_xlabel('h')
ax3D.set_ylabel('s')
ax3D.set_zlabel('amp')
ax3D.view_init(19.9, -34.2)

ax_sinc = fig.add_subplot(gs[ 1, 3:])
ax_sinc.set_xlabel('time $[d^{-1}]$')
ax_sinc.set_ylabel('h(t) $[H_0]$')
ax_sinc.set_title('Timeseries')

ax_heat = fig.add_subplot(gs[ 2, 3])
ax_heat.set_xlabel('h')
ax_heat.set_ylabel('s')
ax_heat.set_title('Heatmap')

ax_regs = fig.add_subplot(gs[ 2, 4])
ax_regs.set_xlabel('s')
ax_regs.set_ylabel('amp')
ax_regs.set_title('S part')
ax_regs.grid()

ax_table = fig.add_subplot(gs[0, 3:])
ax_table.axis('off')

### Plot 2D data

# in 3D axis
tf = t[-1]
N = len(t)
dt = tf/N
estimated_period = 2 * (tau + 1)
plot_range = slice(int(-10*estimated_period/dt), N)

ax3D.plot(h:=data[0][plot_range], s:=data[5][plot_range], 
          reg_func(s, h, sigma, alpha, eta),
          'k', lw=1)

ax3D.plot(h2:=data[6][plot_range], s2:=data[11][plot_range],
          reg_func(s2, h2, sigma, alpha, eta),
          'r', lw=1)

h_lim = np.round(ax3D.get_xlim(), 1)
s_lim = np.round(ax3D.get_ylim(), 1)

# in regulation axis
ax_regs.plot(s, reg_func(s, h, sigma, alpha, eta), label='s part 1', lw=2)
ax_regs.plot(s2, reg_func(s2, h2, sigma, alpha, eta), label='s part 2', lw=2)
# ax_regs.legend(loc='lower right')

# in heatmap axis
ax_heat.plot(h, s, 'k', lw=1)
ax_heat.plot(h2, s2, 'r', lw=1)

# in timeseries axis
ax_sinc.plot(tt:=t[plot_range], h, label='H1')
ax_sinc.plot(tt, data[6][plot_range], '--', label='H2')
legend = None

# in table axis
vals = [f'{v:.2f}' for v in some_s.values]
CI_val = [vals.pop(-2), vals.pop()]
vals.append(CI_val)
param_strings = np.array([f'${c}$={v}' for c, v in zip(runs.columns, vals)])

hstep, vstep, hoffset, voffset = 0.2, 0.2, 0.01, 0.1
x_vals = [hoffset + i*hstep for i in range(5)]
y_vals = [voffset + i*vstep for i in range(3)]
y_vals.reverse()
xy_vals = product(y_vals, x_vals)
for text, (y,x) in zip(param_strings, xy_vals):
    ax_table.text(x, y, text, fontsize=13)

def update_plot(h_lim, s_lim):
    # creat the plot according to the data, but no bigger than the limit values    
    H = np.linspace( h_lim[0], h_lim[1], 70)
    S = np.linspace( s_lim[0], s_lim[1], 70)
    H, S = np.meshgrid(H[H>0], S[S>0])
    r = reg_func(S, H, sigma, alpha, eta)
        
    ax3D.set_xlim( *h_lim )
    ax3D.set_ylim( *s_lim )
    plots[0] = ax3D.plot_surface(H, S, r, cmap='viridis', edgecolor='none', alpha=0.8)
    
    plots[1] = ax_heat.imshow(r, 
          extent=(*h_lim, *s_lim), 
          aspect='auto',
          origin='lower',
          cmap = 'viridis',
          interpolation = 'none'
          )
    ax_heat.set_xlim(*h_lim)
    ax_heat.set_ylim(*s_lim)
    
    plots[2], = ax_regs.plot(S[:, 0], s_reg_func(S[:, 0], sigma, alpha, eta), '--', label='full', color='C2')
    ax_regs.set_xlim(*s_lim)
    if legend is None:
        # ax_regs.legend()
        pass

plots = [None]*3
update_plot(h_lim, s_lim)

def check_number(textbox):
    try:
        number = float(textbox.text)
        textbox.color = '.95'
        textbox.hovercolor = '.8'
        return number
    except ValueError:
        textbox.color = 'xkcd:salmon'
        textbox.hovercolor = 'xkcd:dark salmon'

def on_submit(text):
    limits = [
        check_number(text_hmin),
        check_number(text_hmax),
        check_number(text_smin),
        check_number(text_smax),
        ]
    if all(l is not None for l in limits):
        for x in plots:
            x.remove()
        s, h, r = update_plot(limits[:2], limits[2:])
        plots
        
axbox_hmin = plt.axes([0.05, 0.15, 0.05, 0.025])
text_hmin = TextBox(axbox_hmin, 'min', initial=h_lim[0], hovercolor='.8')
text_hmin.on_submit(on_submit)

axbox_hmax = plt.axes([0.15, 0.15, 0.05, 0.025])
text_hmax = TextBox(axbox_hmax, 'max', initial=h_lim[1], hovercolor='.8')
text_hmax.on_submit(on_submit)

axbox = plt.axes([0.05, 0.05, 0.05, 0.025])
text_smin = TextBox(axbox, 'min', initial=s_lim[0], hovercolor='.8')
text_smin.on_submit(on_submit)

axbox = plt.axes([0.15, 0.05, 0.05, 0.025])
text_smax = TextBox(axbox, 'max', initial=s_lim[1], hovercolor='.8')
text_smax.on_submit(on_submit)

fig.text(0.08, 0.19, 'h limits', fontweight='bold', fontsize=12)
fig.text(0.08, 0.09, 's limits', fontweight='bold', fontsize=12)

#### ADD parameter table