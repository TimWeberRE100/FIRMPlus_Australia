# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:48:18 2024

@author: u6942852
"""

import datetime as dt
import csv
import numpy as np
from numba import njit


from Input import *
from optimisation_utils import Jacobian


@njit 
def objective(x):
    S = Solution(x)
    S._evaluate()
    return S.Lcoe


def normalise(x):
    return (x-lb)/(ub-lb)

def unnormalise(x):
    return x*(ub-lb) + lb

if __name__=='__main__':
    #If we remove HVDC loss term from objective function then we linearise the cost function
    
    
    
    x0 = unnormalise(np.random.rand(len(lb)))
    convex = True
    counter = 0
    while convex:
        x1 = unnormalise(np.random.rand(len(lb)))
        
        # https://math.stackexchange.com/questions/3325382/how-to-check-if-a-function-is-convex
        f_ave = objective((x0 + x1)/2)
        
        if convex:= (f_ave <= ( (objective(x0) + objective(x1))/2 )):
            counter+=1 
            x0 = x1.copy()
        else: 
            print('non-convex')
        if counter% 10 == 0:
            print(counter, ',', end = '')
        
    
