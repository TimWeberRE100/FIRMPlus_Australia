# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:45:41 2024

@author: u6942852
"""

import numpy as np 
import csv
import warnings
from datetime import datetime
from Input import *

class working_solution:
    def __init__(self, func, base_x, inc, ub, lb, convex=None):
        self.objective = func
        self.base_x = base_x
        self.inc = inc
        self.lb, self.ub = lb, ub
        self.it = 0
        
        self = self.update_self(base_x, inc)

    def update_self(self, base_x, inc, j=None, it=None):
        self.base_x = base_x
        self.inc = inc
        self.base_obj = self.objective(base_x)
        
        self.j = j
        self.it = it
        
        if j is not None:  
            self.p_obj, self.p_x = self.eval_sample(base_x, inc, j)
            self.n_obj, self.n_x = self.eval_sample(base_x, -inc, j)
            
            self.p_step = (self.p_obj - self.base_obj)
            self.n_step = (self.n_obj - self.base_obj)
            
            self.p_grad = self.p_step/self.inc
            self.n_grad = self.n_step/self.inc

            if (self.p_grad < 0) and (self.n_grad < 0):
                self.p_grad = 0 if self.p_grad > self.n_grad else self.p_grad
                self.n_grad = 0 if self.p_grad < self.n_grad else self.n_grad
        
        return self

    def eval_sample(self, base_x, inc, i):
        samp_x = base_x.copy()
        samp_x[i] += inc
        if inc > 0: 
            samp_x = np.clip(samp_x, None, self.ub)
        else:
            samp_x = np.clip(samp_x, self.lb, None)
        
        return self.objective(samp_x), samp_x
        

def local_sampling(
        func, 
        x0, 
        bounds=None,
        maxiter=1000,
        disp=False, 
        callback=None, 
        incs=(10,1,0.1,0.01), 
        convex=True, 
        atol=1e-6, 
        rtol=-np.inf):
    
    def writerow_callback(ws):
        with open(callback, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ws.it, ws.j, ws.base_obj, ws.inc, ws.p_step, ws.n_step] + list(ws.base_x))

    def init_callbackfile(n):
        with open(callback, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['iteration', 'dvar index', 'objective', 'increment', 'pos step obj', 'neg step obj'] + ['dvar']*n)
    
    if bounds is not None:
        lb, ub = zip(*((pair for pair in bounds)))
        lb, ub = np.array(lb), np.array(ub)
        assert (x0 < lb).sum() == (x0 > ub).sum() == 0, "starting point outside of bounds"
    else:
        lb, ub = None, None
    

    
    base_x = x0.copy()
    ii, i = 0, 0
    inc = incs[ii]
    
    ws = working_solution(func, base_x, inc, ub, lb, convex)
    
    if disp is True:
        print(f"Optimisation starts: {datetime.now()}\n{'-'*40}")
    
    while i < maxiter:
        if disp is True: 
                print(f"iteration {i}: {ws.base_obj}")
        base_obj=ws.base_obj
        for j in range(len(base_x)):
            ws.update_self(base_x, inc, j, i)
            
            if callback is not None: 
                writerow_callback(ws)
            
            if ws.p_grad < 0: 
                base_x[j] += inc
                base_x = np.clip(base_x, lb, ub)
            elif ws.n_grad < 0:
                base_x[j] -= inc
                base_x = np.clip(base_x, lb, ub)

        dif = abs(ws.base_obj - base_obj) 
        if dif < atol or dif/base_obj < rtol:
            try: 
                ii+=1
                inc = incs[ii]
            except IndexError:
                termination = "Reached finest increment resolution"
                break
        i+=1

    if i == maxiter:
        termination = "Reached maximum iterations"
        
    return ws, termination

def Objective(x):
    """This is the objective function"""
    S = Solution(x)
    S._evaluate()
    return S.Lcoe + S.Penalties

if __name__ == '__main__':
    x0 = np.array([1.0446808145873447,1.9225514380927677,4.547934821604063,0.6394797800384957,
                   1.7059832627565292,5.567044483912056,0.05223608397662716,2.1971672144528327,
                   3.259704305018225,0.4193800897525506,2.6071781764238686,2.2934698861931935,
                   2.76901161945241,1.9983443884383547,1.960225220747212,1.0415666839054332,
                   1.8738282484788051,2.2524137002835296,1.1010883667387468,0.6986379302954262,
                   1.0224143771140497,0.6468666022634082,0.21088020419054487,3.3072768689228305,
                   0.6064128476638757,8.094702782301585,0.7392077463568434,0.30922617218560333,
                   1.9982532504203512,1.5194890690330745,1.384427306563456,0.84969485083964,
                   2.169773084778093,0.6735672385524509,0.4925133963126598,0.439554306233056,
                   0.6727742164504846,0.2673883696422905,3.5355396852215932,2.1942385990682354,
                   0.3504616689687836,1.4629381155526993,10.905055741962192,7.652908907343964,
                   1.0820017344681965,2.1194557171036728,4.013884301951791,66.57163040293744,
                   128.07654060333152,4.847566298809397,1.622335387779458,117.33265359235293,
                   7.470250276232303,9.100887995000262,4.560521627477144,3.1972631971862384])
    
    if args.cb >= 1: 
        cbfile = 'Results/GOptimHistory{}.csv'.format(scenario) 
    else:
        cbfile = None

    ws, termination = local_sampling(
        func=Objective,
        x0=x0,        
        bounds=list(zip(lb, ub)), 
        maxiter=50,
        disp=True,
        incs=[(10**n) for n in range(1, -6, -1)],
        callback=cbfile,
        convex=None,
        )

    print(termination)