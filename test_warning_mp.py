#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:11:34 2021

@author: user
"""

import warnings
import multiprocessing as mp
from time import sleep

def first():
    warnings.warn('Error in 1', UserWarning)
    
    sleep(5)
    print('Exited 1')
        
def second():
    warnings.warn('Warning in 2', UserWarning)
    print('Exited 2')

if __name__ == '__main__':
    
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        
        mp.Process(target=first).start()
        mp.Process(target=second).start()
