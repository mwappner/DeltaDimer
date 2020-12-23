# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:32:07 2020

@author: Marcos
"""

from ddeint import ddeint
import numpy as np

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
               lm, lp, kC, kD, kE): #l == lambda, k == kappa, lm == lm/H0
    
    h1, c1, d1, e1, n1, s1, h2, c2, d2, e2, n2, s2 = X(t)
    h1d, _, _, _, _, s1d, h2d, _, _, _, _, s2d = X(t-tau)
    
    model = [
        -h1 + b_H * (1 + s*a * s1d**eta ) / ( (1 + s * s1d**eta) * (1 + h1d**eta) ),
        -c1 + b_C / (1 + h1d**eta) + lm * e1 - lp * c1 * d1 - kC * c1 * n2,
        -d1 + b_D + lm * e1 - lp * c1 * d1 - kD * d1 * n2,
        -e1 - lm * e1 + lp * c1 * d1 - kE * e1 * n2,
        -n1 + b_N - n1 * (kC * c2 + kD * d2 + kE * e2),
        -s1 +  n1 * (kC * c2 + kD * d2 + kE * e2),
        
        -h2 + b_H * (1 + s*a * s2d**eta ) / ( (1 + s * s2d**eta) * (1 + h2d**eta) ),
        -c2 + b_C / (1 + h2d**eta) + lm * e2 - lp * c2 * d2 - kC * c2 * n1,
        -d2 + b_D + lm * e2 - lp * c2 * d2 - kD * d2 * n1,
        -e2 - lm * e2 + lp * c2 * d2 - kE * e2 * n1,
        -n2 + b_N - n2 * (kC * c1 + kD * d1 + kE * e1),
        -s2 +  n2 * (kC * c1 + kD * d1 + kE * e1),
        ]
    return np.array(model)

times = np.linspace(0, 10, 100)

d = 0.3 # 1/min
b_H, b_C, b_D, b_N = [30]*4
kCDp, kCDm = 0.01, 0.1
kCN, kDN, kEN = 0.1, 0.1, 0.1
H0 = 1 #!!!

parameters = {
    'tau' : 10        ,
    'b_H' : b_H/d/H0  ,
    'b_C' : b_C/d/H0  ,
    'b_D' : b_D/d/H0  ,
    'b_N' : b_N/d/H0  ,
    'eta' : 2.3       ,
    'a'   : 1         ,
    's'   : 1         ,
    'lm'  : kCDm/d    , # lm == lm/H0
    'lp'  : kCDp*H0/d ,
    'kC'  : kCN*H0/d  ,
    'kD'  : kDN*H0/d  ,
    'kE'  : kEN*H0/d  ,
    }

def past_values(t):
    return np.ones(12)

Xint = ddeint(model_adim, past_values, times, fargs=parameters.values())