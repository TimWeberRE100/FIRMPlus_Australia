# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import datetime as dt
import csv
import numpy as np
from argparse import ArgumentParser
from multiprocessing import cpu_count, Pool
from scipy.optimize import differential_evolution

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=21, type=int, required=False, help='11, 12, 13, ...')
parser.add_argument('-cb', default=0, type=int, required=False, help='Callback: 0-None, 1-generation elites, 2-everything')
parser.add_argument('-ver', default=1, type=int, required=False, help='Boolean - print progress to console')
parser.add_argument('-vp', default=50, type=int, required=False, help='Maximum number of vectors to send to objective')
parser.add_argument('-w', default=1, type=int, required=False, help='Maximum number of cores to parallelise over')
parser.add_argument('-vec', default=1, type=int, required=False, help='Boolean - vectorised mode')

args = parser.parse_args()

assert args.w > 0 or args.w in (-1, -2)

scenario = args.s

from Input import *
    
def Vobjective(x):
    """Vectorised Objective Function"""
    S = VSolution(x)
    S._evaluate()
    return S.Lcoe + S.Penalties

def objective(x):
    """This is the objective function"""
    S = Solution(x)
    S._evaluate()
    return S.Lcoe + S.Penalties

def vobj_wrapper(x, callback):
    
    arrs = [x[:, n*vsize: min((n+1)*vsize, npop)] for n in r]

    if processes > 1:
        with Pool(processes=processes) as processPool:
            results = processPool.map(Vobjective, arrs)
        results = np.concatenate(results)
    else:
        results = np.concatenate([Vobjective(arr) for arr in arrs])
    
    if callback is True: 
        printout = np.concatenate((results.reshape(-1, 1), x.T), axis = 1)
        with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(printout)
    
    return results

def callback(xk, convergence=None):
    with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([objective(xk)] + list(xk))
        
def init_callback():
    with open('Results/History{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

def Optimise(args=args):
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    if args.cb > 1: 
        init_callback()

    if args.vec == 1: 
        func = vobj_wrapper
        func_args = (args.cb==2,)
    elif args.vec == 0:
        func = objective
        func_args = ()

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
        callback=callback if args.cb == 1 else None,
        workers=args.w if args.vec == 0 else 1,
        )
    
    endtime = dt.datetime.now()
    timetaken = endtime-starttime
    print("Optimisation took", timetaken)

    return result, timetaken

if __name__=='__main__':

    npop = args.p * len(lb)
    processes = cpu_count() if args.w == -1 else cpu_count()//2 if args.w ==-2 else args.w
    processes = min(npop, processes)
    
    if bool(args.vec) is False:  
        vsize = 1
    else: 
        vsize = npop//processes + 1 if npop%processes != 0 else npop//processes
        vsize = min(vsize, args.vp, npop)
    r = range(npop//vsize + 1) if npop%vsize != 0 else range(npop//vsize)
    #TODO 
    # make vsize smaller if no. slices is only a few larger than no. processes
    # this will reduce load on each process and avoid waiting for just one extra process 
    
    result, time = Optimise()
    
    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)
    
    
    # from Dispatch import Analysis
    # Analysis(result.x)


