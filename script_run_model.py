#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 23:27:41 2021

@author: user
"""

from DeltaDimer_model_one import integrate
from itertools import product

CIs = [x for x in product(range(10), range(10)) if x[0]>=x[1]]

for CI in CIs:
    integrate(CI)