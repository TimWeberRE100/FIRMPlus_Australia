# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import datetime as dt
import csv
import numpy as np
from multiprocessing import cpu_count, Pool
from scipy.optimize import differential_evolution

from Input import *

def Objective(x, callback=False):
    """This is the objective function"""
    S = Solution(x)
    S._evaluate()
    if callback is True: 
        with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow([S.Lcoe+S.Penalties] + list(x))
    return S.Lcoe + S.Penalties

def Callback_1(xk, convergence=None):
    with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
        csv.writer(csvfile).writerow([Objective(xk)] + list(xk))
        
def Init_callback():
    with open('Results/History{}.csv'.format(scenario), 'w', newline='') as csvfile:
        csv.writer(csvfile)

def Optimise():
    
    Objective(np.random.rand(len(ub))*(ub-lb)+lb) #compile jit

    if args.cb > 1: 
        Init_callback()

    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)
    result = differential_evolution(
        func=Objective, 
        args=(args.cb==2,),
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=args.m, 
        recombination=args.r,
        disp=bool(args.ver), 
        polish=False, 
        updating='deferred', 
        callback=Callback_1 if args.cb == 1 else None,
        workers=args.w, 
        )
    
    endtime = dt.datetime.now()
    timetaken = endtime-starttime
    print("Optimisation took", timetaken)

    return result, timetaken
    

if __name__=='__main__':

    result, time = Optimise()
    
    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)
    
    
    # from Dispatch import Analysis
    # Analysis(result.x)


