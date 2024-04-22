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


from Optimisation import args#, Vobj_wrapper, Objective
from Input import * 

VERBOSE=True
SLACK = 1.05
# SLACK = 1e9 #for testing
MAXITER=300


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
        if now - self.start > best_time * SLACK:
            if self.disp is True:
                print(f"Iteration {self.it}. Best value: {f}.\nEnding optimisation (5 % longer than current best)")
            # abort if optimisation takes longer than current best
            return True
        if self.it >= MAXITER:
            if self.disp is True:
                print(f"Iteration {self.it}. Best value: {f}.\nEnding optimisation (max iterations, unconverged)")
            # abort if optimisation takes too many iterations
            return True
        return False
        
def Optimise(p, m, r, iter=MAXITER):   
        
    start_time = dt.datetime.now()
    
    result = differential_evolution(
        func=func, 
        args=func_args,
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=iter,
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

def gen_params(npop, processes):
    vsize = npop//processes + 1 if npop%processes != 0 else npop//processes
    vsize = min(vsize, args.vp, npop)
    range_gen = range(npop//vsize + 1) if npop%vsize != 0 else range(npop//vsize)
    ind_pairs = [(n*vsize, min((n+1)*vsize, npop)) for n in range_gen]
    
    return (False, ind_pairs)

def init_writefile():
    with open(filename, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerow(['time', 'p', 'm', 'r', 'obj', 'comment'])
    return True

def writefile(row):
    with open(filename, mode='a', newline='') as csvfile:
        csv.writer(csvfile).writerow(row)
    return True
        
def get_comment(result, time):
    if result.success == True:
        return "locally stuck"
    if Objective(result.x) <= cost:
        return "approximate convergence"
    if result.nit >= MAXITER:
        return "too many iterations"
    if time >= best_time * SLACK:
        return "time longer than best"

def continue_criteria(args, records_args):
    for row in record_args:
        if (args[0] == row[0] and
            args[1] == row[1] and 
            args[2] == row[2]):
            return True
    return False
    

if __name__ == '__main__':    
    filename = 'Results/HyperParameterTuning{}-{}.csv'.format(
            scenario, 0 if bool(args.vec) else cpu_count() if args.w == -1 else args.w)
    
    if args.vec == 0: 
        func = Objective
        func_args = ()
        func(np.random.rand(len(ub))*(ub-lb)+lb) #pre-compile jit
    
    else: #args.vec==1
        func = Vobj_wrapper
        func((np.random.rand(len(ub))*(ub-lb)+lb).reshape(-1,1), False, [(0,1)]) #pre-compile jit
        processes = 1
        func_args = gen_params(10*len(lb), processes)
        
    best_time = dt.timedelta(days=1) # ~= inf 
    record_times = np.array([]).reshape((0,1))
    record_args = np.array([]).reshape((0,4))
    
    try: 
        from pandas import read_csv, to_timedelta
        record = read_csv(filename)
        record_times = to_timedelta(record['time']).to_numpy().astype(np.timedelta64).reshape(-1,1)
        record_args = record[['p','m','r','obj']].to_numpy()
        comments = record['comment']
        mask = comments.isin(['baseline', 'approximate convergence'])
        
        best_time = dt.timedelta(microseconds = int(record_times[mask,:].min()/1000))
        cost = record_args.reshape(-1, 4)[0,3] * SLACK
        
    except FileNotFoundError:
        init_writefile()
        cost = -np.inf
        result, best_time = Optimise(10, 0.5, 0.3)
        cost = Objective(result.x)*SLACK
        writefile([best_time, -1, -1, -1, Objective(result.x), 'baseline'])

    
    for p in range(6, 106, 10):
        if bool(args.vec) is True: 
            func_args = gen_params(p*len(lb), processes)
            
        for m in range(1, 21, 2):
            m /= 10
            for r in range(1, 11, 1):
                r /= 10
                if continue_criteria([p, m, r], record_args[:,:3]) is True:
                    continue
                
                start_time = dt.datetime.now()
                result, time_taken = Optimise(p, m, r)
                
                comment = get_comment(result, time_taken)
                
                writefile([time_taken, p, m, r, Objective(result.x), comment])
                
                record_times = np.vstack((record_times, [time_taken]))
                record_args = np.vstack((record_args, [p, m, r, Objective(result.x)]))
                if time_taken < best_time and result.success is False: 
                    best_time = time_taken
                    
    reindex = record_times.argsort()
    record_times, record_args = record_times[reindex], record_args[reindex] 
    
    print(f"""
Scenario: {scenario}. Vectorisation: {bool(args.vec)}. Workers: {args.w if args.vec == 0 else 1}.
Best time: {record_times[0]} achieved with p, m, r = {record_args[0]}
Second: {record_times[1]} achieved with p, m, r = {record_args[1]}
Third: {record_times[2]} achieved with p, m, r = {record_args[2]}
Full dataset is printed out to Results/HyperParameterTuning{scenario}-{0 if bool(args.vec) else args.w}.csv
          """)
    
    # with open('Results/HyperParameterTuning{}-{}.csv'.format(scenario, 0 if bool(args.vec) else args.w), 'w', newline='') as csvfile:
    #     printout = np.hstack((record_times, record_args))
    #     writer = csv.writer(csvfile)
    #     writer.writerows(printout)
