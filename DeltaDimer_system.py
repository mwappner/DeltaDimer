# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:37:39 2020

@author: Marcos
"""
import sympy as sp
sp.init_printing(use_unicode=True)

c1, c2, s1, s2, d1, d2, e1, e2, n1, n2 = sp.symbols(
        'c_1, c_2, s_1, s_2, d_1, d_2, e_1, e_2, n_1, n_2', nonnegative=True)
bd, bn, kc, kd, ke, lp, lm = sp.symbols(
        'beta_d, beta_n, kappa_c, kappa_d, kappa_e, lambda_+, lambda_- ', real = True)

S21 = kc * c1 + kd * d1 + ke * e1

exp_s2 = (bn * S21)/(1+S21)
exp_d1 = (bd + lm * (lm + ke * n2)) / (1 + lp * c1 * (1-lm) + kd * n2)
exp_e1 = lp * c1 * bd / (lp * c1 * (ke * n2 + 1) + (kd * n2 + 1) * (kd * n2 + lm + 1))
exp_n1 = bn - s1

subs_list = [(c1, c2), (d1, d2), (e1, e2), (n2, n1)]
exp_s1 = exp_s2.subs(subs_list)
exp_d2 = exp_d1.subs(subs_list)
exp_e2 = exp_e1.subs(subs_list)
exp_n2 = exp_n1.subs(s1, s2)

# solution = sp.solve(
#         [s1-exp_s1, s2-exp_s2, d1-exp_d1, d2-exp_d2, e1-exp_e1, e2-exp_e2, n1-exp_n1, n2-exp_n2],
#         [s1, s2, d1, d2, e1, e2, n1, n2]
#         )

exp_d1 = exp_d1.subs(n2, exp_n2)
exp_d1 = exp_d1.subs(s2, exp_s2).simplify()

exp_e1 = exp_e1.subs(n2, exp_n2)
exp_e1 = exp_e1.subs(s2, exp_s2).simplify()

solution = sp.nonlinsolve([e1-exp_e1, d1-exp_d1], [e1, d1])


#%% System of equations

ec1 = c2*c1-bd
ec2 = c2 + bn * c1**2

solution = sp.nonlinsolve([ec1, ec2], [c1, c2])
