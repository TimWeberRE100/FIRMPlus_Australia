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

def Vobjective(x):
    """Vectorised Objective Function"""
    S = VSolution(x)
    S._evaluate()
    return S.Lcoe + S.Penalties

def Vobj_wrapper(x, callback, ind_pairs):
    arrs = [x[:, i: j] for i, j in ind_pairs]
    results = np.concatenate([Vobjective(arr) for arr in arrs])

    if callback is True:
        # with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
        #     printout = np.concatenate((results.reshape(-1, 1), x.T), axis = 1)
        #     writer = csv.writer(csvfile)
        #     writer.writerows(printout)
        csvfile = open('Results/History{}.csv'.format(scenario), 'a', newline='')
        printout = np.concatenate((results.reshape(-1, 1), x.T), axis = 1)
        csv.writer(csvfile).writerows(printout)
        csvfile.close()
    return results

def Objective(x):
    """This is the objective function"""
    S = Solution(x)
    S._evaluate()
    return S.Lcoe + S.Penalties

def Vobj_mpwrapper(x, callback, ind_pairs, processes):
    arrs = [x[:, i: j] for i, j in ind_pairs]
    with Pool(processes=processes) as processPool:
        results = processPool.map(Vobjective, arrs)
        results = np.concatenate(results)
    
    if callback is True:
        with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
            printout = np.concatenate((results.reshape(-1, 1), x.T), axis = 1)
            writer = csv.writer(csvfile)
            writer.writerows(printout)
    return results

def Callback_1(xk, convergence=None):
    with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
        csv.writer(csvfile).writerow([Objective(xk)] + list(xk))
        
def Init_callback():
    with open('Results/History{}.csv'.format(scenario), 'w', newline='') as csvfile:
        csv.writer(csvfile)

def Optimise():
    if args.cb > 1: 
        Init_callback()

    if args.vec == 1: 
        npop = args.p * len(lb)
        processes = cpu_count() if args.w == -1 else cpu_count()//2 if args.w ==-2 else args.w
        processes = min(npop, processes)
        
        if processes == 1:
            func = Vobj_wrapper
        elif processes > 1:
            func = Vobj_mpwrapper
            
        vsize = npop//processes + 1 if npop%processes != 0 else npop//processes
        vsize = min(vsize, args.vp, npop)
        range_gen = range(npop//vsize + 1) if npop%vsize != 0 else range(npop//vsize)
        ind_pairs = [(n*vsize, min((n+1)*vsize, npop)) for n in range_gen]
        
        if processes == 1:
            func_args = (args.cb==2, ind_pairs)
        elif processes > 1:
            func_args = (args.cb==2, ind_pairs, processes)
        
    elif args.vec == 0:
        vsize=1
        func = Objective
        func_args = ()
        
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)
    result = differential_evolution(
        func=func, 
        args=func_args,
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=args.m, 
        recombination=args.r,
        disp=bool(args.ver), 
        polish=False, 
        updating='deferred', 
        vectorized=bool(args.vec),
        callback=Callback_1 if args.cb == 1 else None,
        workers=args.w if args.vec == 0 else 1, #vectorisation overrides mp
        )
    
    endtime = dt.datetime.now()
    timetaken = endtime-starttime
    print("Optimisation took", timetaken)

    return result, timetaken

if __name__=='__main__':


    
    #TODO 
    # make vsize smaller if no. slices is only a few larger than no. processes
    # this will reduce load on each process and avoid waiting for just one extra process 
    
    result, time = Optimise()
    
    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)
    
    
    # from Dispatch import Analysis
    # Analysis(result.x)


