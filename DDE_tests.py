# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 23:32:54 2020

@author: Marcos
"""

import sympy as sp
x, y, z = sp.symbols('x y z')
sp.init_printing()

#%%
t, d = sp.symbols('t d')
y = sp.Function('y')
yt = sp.Function('yt')
f = sp.Function('f')(t, y(t), y(t-d))
p = t-d
g = sp.Function('g')(p, y(p), yt(p))


#%%

c2, c3 = sp.symbols('c2 c3')

A2 = (2-3*c3)/(6*c2 * (c2-c3))
A3 = (3*c2-2)/(6*c3 * (c2-c3))
A1 = 1 - A2 - A3

a32 = (c3*(c2-c3))/(c2*(3*c2-2))
a31 = c3 - a32

vals = [(c2, 1/2), (c3, 1)]
A1.subs(vals)
A2.subs(vals)
A3.subs(vals)
a32.subs(vals)
a31.subs(vals)
