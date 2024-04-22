# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:27:19 2024

@author: u6942852
"""


import numpy as np

import datetime as dt

from Input import *
from Optimisation import args#, Vobj_wrapper
from DirectAlgorithm import Direct
from numba import jit, objmode
from csv import writer

@jit(nopython=False)
def Vobj(x, callback=False):
    results = np.empty(len(x.T), dtype=np.float64)
    S = VSolution(x)
    S._evaluate()
    results[:] = S.Lcoe + S.Penalties
    
    if callback is True:
        printout = np.concatenate((results.reshape(-1, 1), x.T), axis = 1)
        with objmode():
            csvfile = open('Results/History{}.csv'.format(scenario), 'a', newline='')
            writer(csvfile).writerows(printout)
            csvfile.close()
    return results
    
def Callback_1(h):
    with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
        writer(csvfile).writerow([h.f] + list(h.centre))
        
def Init_callback():
    with open('Results/History{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer(csvfile)

if __name__ == '__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)
    
    if args.cb > 0: 
        Init_callback()
            
    min_length = np.array([10e-4]*(pzones+wzones+nodes) + [0.1])
    # 1 MW, 100 MWh 
    
    result = Direct(
        func=Vobj, 
        bounds=(lb, ub),
        f_args=(args.cb==2,),
        maxiter=args.i, 
        callback= Callback_1 if args.cb==1 else None, 
        vectorizable=True,
        maxvectorwidth=args.vp,
        population=args.p,
        min_length = min_length, # float or array of x.shape
        rect_dim=4,
        disp = bool(args.ver),
        locally_biased=False,
        )

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    print(result.x, result.f)