# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:17:49 2024

@author: u6942852
"""

import numpy as np
import datetime as dt
from scipy.optimize import differential_evolution
from multiprocessing import cpu_count

from Optimisation import args, Vobj_wrapper, Objective
from Input import * 

VERBOSE=True
SLACK = 1.05
# SLACK = 1e9 #for testing
MAXITER=10000

cost = np.genfromtxt('Data/OptimisationResults.csv', dtype=float, skip_header=1, delimiter=',')
cost = cost[cost[:,0] == args.s, 1][0]
cost = cost*SLACK

class callback:
    def __init__(self, disp_freq=0):
        self.start = dt.datetime.now()
        self.it = 0
        self.disp = disp_freq != 0
        self.disp_freq = disp_freq
    
    def __call__(self, xk, convergence=None):
        f = Objective(xk)
        now = dt.datetime.now()
        if self.disp is True:
            if self.it % self.disp_freq == 0:
                print(f"Iteration {self.it}. Best value: {f}. Average time: {(now-self.start)/(self.it+1)} per it")            
        self.it += 1 
        if f <= cost:
            if self.disp is True:
                print(f"Iteration {self.it}. Best value: {f}.\nEnding optimisation (approximately converged)")            
            # abort when optimisation reaches approximate convergence
            return True
        if now - self.start > best_time + dt.timedelta(minutes=5):
            if self.disp is True:
                print(f"Iteration {self.it}. Best value: {f}.\nEnding optimisation (5 minutes longer than current best)")
            # abort if optimisation takes longer than current best
            return True
        if self.it >= MAXITER:
            if self.disp is True:
                print(f"Iteration {self.it}. Best value: {f}.\nEnding optimisation (max iterations, unconverged)")
            # abort if optimisation takes too many iterations
            return True
        return False
    
# def callback(xk, convergence=None):
#     if Objective(xk) <= cost: 
#         # abort when optimisation reaches approximate convergence
#         return True
#     if dt.datetime.now() - start_time > best_time + dt.timedelta(minutes=5):
#         # abort if optimisation takes longer than current best
#         return True
#     return False
    
def Optimise(p, m, r):       
    start_time = dt.datetime.now()

    result = differential_evolution(
        func=func, 
        args=func_args,
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=MAXITER,
        popsize=p, 
        mutation=m, 
        recombination=r,
        disp=0, 
        polish=False, 
        updating='deferred', 
        vectorized=bool(args.vec),
        callback=callback(100),
        workers=args.w if args.vec == 0 else 1, #vectorisation overrides mp
        )
    
    end_time = dt.datetime.now()
    time_taken = end_time-start_time

    return result, time_taken

if __name__ == '__main__':
    if args.vec == 0: 
        func = Objective
        func_args = ()
    
    best_time = dt.timedelta(days=1) # ~inf
    best_args = (0,0,0)
    
    for p in range(1, 206, 5):
        if bool(args.vec) is True: 
            
            func = Vobj_wrapper
            processes = 1
            npop = p * len(lb)

            vsize = npop//processes + 1 if npop%processes != 0 else npop//processes
            vsize = min(vsize, args.vp, npop)
            
            range_gen = range(npop//vsize + 1) if npop%vsize != 0 else range(npop//vsize)
            ind_pairs = [(n*vsize, min((n+1)*vsize, npop)) for n in range_gen]
            
            func_args = (False, ind_pairs)
            
        for m in range(1, 21, 1):
            m /= 10
            
            for r in range(1,11,1):
                r /= 10
                print(p,m,r, best_time)
                
                start_time = dt.datetime.now()
                result, time_taken = Optimise(p, m, r)
                
                if time_taken < best_time:
                    best_time = time_taken
                    best_args = p, m, r

    
