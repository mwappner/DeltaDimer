# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 00:34:25 2020

@author: Marcos
"""
### https://github.com/neurophysik/jitcdde

from jitcdde import jitcdde, y, t
import numpy
# y, t son símbolos que se importan
# hay que definir el modelo en función de estos para que el integrador funcione
# usa Symoy en le backend


τ = 15
n = 10
β = 0.25
γ = 0.1

# para un sistema N dimensional en general toma una lista
# acá N=1, pero bueno, sigue siendo una lista
f = [ β * y(0,t-τ) / (1 + y(0,t-τ)**n) - γ*y(0) ]
DDE = jitcdde(f)

# hay muchas formas de definir el pasado
# en este caso, quiero pasado constante = 1
DDE.constant_past([1.0])

# por problemas en las condiciones iniciales, y cómo el pasado no
# corresponde con lo que espero que ydot(t=0) valga
# ver https://jitcdde.readthedocs.io/en/stable/#discontinuities para más
DDE.step_on_discontinuities()

# integramos 10000 pasos con sampling rate (??) de 10 pasos
data = []
for time in numpy.arange(DDE.t, DDE.t+10000, 10):
	data.append( DDE.integrate(time) )
# numpy.savetxt("timeseries.dat", data)