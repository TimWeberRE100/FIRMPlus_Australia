# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 08:12:32 2024

@author: u6942852
"""
 
from numba import njit, prange, float64
import numpy as np
import datetime as dt

from Input import *
from Simulation import Reliability    

@njit
def Jacobian(x, lossfactor=None):
    """
    Approximate local gradient of objective function w.r.t. each input variable
    """

    jac = np.zeros((len(x), len(factor)+1), dtype=np.float64)

    jac[:pidx, 0] += factor[0]
    jac[:pidx, 11] += factor[11]
    jac[pidx:widx, 1] += factor[1]
    jac[pidx:widx, 12] += factor[12]
    jac[widx:spidx, 2] += factor[2]
    jac[spidx:seidx, 3] += factor[3]

    network_factors = np.arange(4, 11)[network_mask]
    for i in range(network_mask.sum()):
        jac[seidx+i, network_factors[i]] += factor[network_factors[i]]

    if lossfactor is None:    
        S = Solution(x) 
        Reliability(S, flexible=np.ones((intervals, nodes), dtype=np.float64)*CPeak*1000)
        
        loss = np.zeros(len(network_mask), dtype=np.float64)
        loss[network_mask] = S.TDC.sum(axis=0) * DCloss[network_mask]
        jac[:seidx,-1] = np.abs(energy - loss.sum() * 0.000000001 * resolution / years)
        loss = loss[network_mask] / x[seidx:] * (x[seidx:]+1)
        jac[seidx:,-1] = np.abs(energy - loss.sum() * 0.000000001 * resolution / years)
    
    else: 
        jac[:, -1] = lossfactor
    
    jac = jac[:, :-1].sum(axis=1)/jac[:,-1]
    
    return jac#.sum(axis=1)

@njit
def cost(x):
    S = Solution(x)
    S._evaluate()
    return S.Lcoe

@njit()
def hessian(x, precision=1, dx=0.001, ddx=0.001):
    jac1 = Jacobian(x, precision, dx)
    jac2 = Jacobian(x, precision, dx+ddx)
    hes = (jac2-jac1)/ddx
    return hes
    
@njit(parallel=True)
def direct_jacobian(x, dx = 0.001):
    cost(x) 
    jac = np.empty(len(x), dtype=np.float64)
    for i in prange(len(x)):
        x0 = x.copy()
        x0[i] = max(x0[i]+dx, ub[i])
        x0i = x0[i]
        cost0 = cost(x0)
        x0[i] = min(x0[i]-2*dx, lb[i])
        dist = x0i-x0[i]
        jac[i]  = cost0-cost(x0)/ dist
    return jac

def jac_test(x):
    t_jac = dt.datetime.now()
    jac = Jacobian(x) 
    t_jac = dt.datetime.now() - t_jac
    
    t_dir = dt.datetime.now()
    base = direct_jacobian(x)
    t_dir = dt.datetime.now() - t_dir
        
    dzsm, direc = jac != 0, np.zeros(jac.shape)
    direc[dzsm] = base[dzsm]/jac[dzsm]
    direc = int((direc > 0).sum())
    print(f""" 
Jacobian test (heuristic v. direct measurement), dx = {dx}, precision = {precision}
Correct Direction: {direc}/{len(x)} ({round(100*direc/len(x), 2)} %)
Overestimated: {int((base/jac > 1).sum())}
Underestimated: {int((base/jac < 1).sum())}
Mean error: {round(np.abs(base - jac).sum()/len(x), 2)}
Mean overestimate: {round(np.abs((base-jac)[base/jac > 1]).sum()/ (base/jac > 1).sum(), 2) 
                    if (base/jac > 1).sum() != 0 else np.nan}
Mean underestimate:  {round(np.abs((base-jac)[base/jac < 1]).sum()/ (base/jac < 1).sum(), 2) 
                      if (base/jac < 1).sum() != 0 else np.nan}
Jacobian heuristic time (jit): {t_jac}
Direct Measurement time (jit): {t_dir}

""")
    return jac, base 
        
if __name__ == '__main__':
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)

    np.set_printoptions(suppress=True)
    print(Jacobian(x))
    
    # x = np.genfromtxt('Results/Optimisation_resultx21.csv', dtype=np.float64, delimiter=',')
    
    # jac = Jacobian(x, -1, 0.001) # compile jit
    # jac, base = jac_test(x, 0, 0.001)
    # jac, base = jac_test(x, 1, 0.001)
    # jac, base = jac_test(x, 2, 0.001)
    
    # hes = hessian(x, 1)
