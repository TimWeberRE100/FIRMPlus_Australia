# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:17:49 2024

@author: u6942852
"""

import numpy as np
import datetime as dt
from scipy.optimize import differential_evolution
from multiprocessing import cpu_count
import csv


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
        callback=callback(1),
        workers=args.w if args.vec == 0 else 1, #vectorisation overrides mp
        )
    
    end_time = dt.datetime.now()
    time_taken = end_time-start_time

    return result, time_taken

if __name__ == '__main__':
    if args.vec == 0: 
        func = Objective
        func_args = ()
        func(np.random.rand(len(ub))*(ub-lb)+lb) #pre-compile jit
    
    else: #args.vec==1
        func = Vobj_wrapper
        func((np.random.rand(len(ub))*(ub-lb)+lb).reshape(-1,1), False, [(0,1)]) #pre-compile jit
        processes = 1
    
    best_time = dt.timedelta(days=1) # ~= inf 
    record_times = np.array([]).reshape((0,1))
    record_args = np.array([]).reshape((0,3))
    
    for p in range(1, 106, 5):
        print('p', p, dt.datetime.now())
        if bool(args.vec) is True: 
            npop = p * len(lb)
            vsize = npop//processes + 1 if npop%processes != 0 else npop//processes
            vsize = min(vsize, args.vp, npop)
            range_gen = range(npop//vsize + 1) if npop%vsize != 0 else range(npop//vsize)
            ind_pairs = [(n*vsize, min((n+1)*vsize, npop)) for n in range_gen]
            
            func_args = (False, ind_pairs)
            
        for m in range(1, 21, 1):
            print('m', m, dt.datetime.now())
            m /= 10
            for r in range(1, 11, 1):
                r /= 10
                start_time = dt.datetime.now()
                result, time_taken = Optimise(p, m, r)
                
                record_times = np.vstack((record_times, [time_taken]))
                record_args = np.vstack((record_args, [p, m, r]))
                if time_taken < best_time: 
                    best_time = time_taken

                    
    reindex = record_times.argsort()
    record_times, record_args = record_times[reindex], record_args[reindex] 
    
    print(f"""
Scenario: {scenario}. Vectorisation: {bool(args.vec)}. Workers: {args.w if args.vec == 0 else 1}.
Best time: {record_times[0]} achieved with p, m, r = {record_args[0]}
Second: {record_times[1]} achieved with p, m, r = {record_args[1]}
Third: {record_times[2]} achieved with p, m, r = {record_args[2]}
Full dataset will be printed out to Results/HyperParameterTuning{scenario}-{0 if bool(args.vec) else args.w}.csv
          """)
    
    with open('Results/HyperParameterTuning{}-{}.csv'.format(scenario, 0 if bool(args.vec) else args.w), 'w', newline='') as csvfile:
        printout = np.hstack((record_times, record_args))
        writer = csv.writer(csvfile)
        writer.writerows(printout)
