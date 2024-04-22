# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 08:12:32 2024

@author: u6942852
"""
 
from numba import njit, boolean
import numpy as np
import datetime as dt

from Input import *
from Simulation import Reliability
from Network import Transmission

#%%

@njit() 
def _cdc(TDC):
    CDC = np.zeros(len(DCloss), dtype=np.float64)
    for j in range(len(DCloss)):
        for i in range(intervals):
            CDC[j] = np.maximum(CDC[j], TDC[i, j])
    return CDC * 0.001 # CDC(k), MW to GW
    

@njit()
def Jacobian(x, precision=1, dx=0.001):
    """
    Approximate local gradient of objective function w.r.t. each input variable
    
    precision 0  - just accounts for infrastructure costs (only option necessary 
                                                          for scenario <= 17)
    precision 1  - accounts for infrastructure costs and some transmission use
                   difference 
    precision 2  - recalculates flexible energy use 
    precision -1 - directly measures by evaluating objective
    
    each precision step increases accuracy but increases compute time. 
    For scenario >= 21, precision 1 is a good balanced option.
    """

    jac = np.zeros((len(x), len(factor)+1), dtype=np.float64)

    if precision >= 0: 
        jac[:pidx, 0] += factor[0]
        jac[:pidx, 11] += factor[11]
        jac[pidx:widx, 1] += factor[1]
        jac[pidx:widx, 12] += factor[12]
        jac[widx:sidx, 2] += factor[2]
        jac[sidx, 3] += factor[3]
 
    if precision >= 1 and scenario >= 21:
        S = Solution(x)
        Reliability(S, flexible=np.ones((intervals, ), dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
        TDC = np.abs(Transmission(S)) if scenario>=21 else np.zeros((intervals, len(DCloss)), dtype=np.float64)
        CDC = _cdc(TDC)
        cdc_factor = factor[4:11]
        l_factor =  0.000000001 * resolution / years 
        for z in range(pzones):
            dxfactor = (1+dx/x[z])
            S.GPV[:, z] *= dxfactor
            if precision >= 2:
                Reliability(S, flexible=np.ones((intervals, ), dtype=np.float64)*CPeak.sum()*1000)
            TDC = np.abs(Transmission(S))
            cdc = _cdc(TDC)
            S.GPV[:, z] /= dxfactor
            
            jac[z, 4:11] += ((cdc-CDC)*cdc_factor) / dx
            jac[z, -1] += (TDC.sum(axis=0) * DCloss).sum(axis=0) * l_factor
                
        for z in range(wzones):
            dxfactor = (1+dx/x[z+pidx])
            S.GWind[:, z] *= dxfactor
            if precision >= 2: 
                Reliability(S, flexible=np.ones((intervals, ), dtype=np.float64)*CPeak.sum()*1000)
            TDC = np.abs(Transmission(S))
            cdc = _cdc(TDC)
            S.GWind[:, z] /= dxfactor
            
            jac[z+pidx, 4:11] += ((cdc-CDC)*cdc_factor) / dx
            jac[z+pidx, -1] += (TDC.sum(axis=0) * DCloss).sum(axis=0) * l_factor
            
        for z in range(nodes): 
            S.CPHP[z] += dx
            if precision >= 2:
                Reliability(S, flexible=np.ones((intervals, ), dtype=np.float64)*CPeak.sum()*1000)
            TDC = np.abs(Transmission(S))
            cdc = _cdc(TDC)
            S.CPHP[z] -= dx
            
            jac[z+widx, 4:11] += ((cdc-CDC)*cdc_factor) / dx
            jac[z+pidx, -1] += (TDC.sum(axis=0) * DCloss).sum(axis=0) * l_factor

        dx *= 100
        S.CPHS += dx
        if precision >= 2:
            Reliability(S, flexible=np.ones((intervals, ), dtype=np.float64)*CPeak.sum()*1000)
        TDC = np.abs(Transmission(S))
        cdc = _cdc(TDC)
        S.CPHS -= dx
        
        jac[-1, 4:11] += ((cdc-CDC)*cdc_factor) / dx
        jac[-1, -1] += (TDC.sum(axis=0) * DCloss).sum(axis=0) * l_factor

    if precision >= 1: 
        return jac[:, :-1].sum(axis=1) / np.abs(energy - jac[:,-1])
    elif precision >= 0: 
        return jac[:, :-1].sum(axis=1)
    
    if precision < 0:
        jac_1 = np.zeros(x.shape)
        S = Solution(x)
        S._evaluate()
        base = S.Lcoe
        for z in range(len(x)):
            if z == len(x) - 1:
                dx *= 100
            x1 = x.copy()
            x1[z] += dx
            S = Solution(x1)
            S._evaluate()
            jac_1[z] += (S.Lcoe - base)/dx
    
        return jac_1
    raise Exception

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
    

def jac_test(x, precision=1, dx=0.001):
    t_jac = dt.datetime.now()
    jac = Jacobian(x, precision, dx) 
    t_jac = dt.datetime.now() - t_jac
    
    t_dir = dt.datetime.now()
    base = Jacobian(x, -1, dx) 
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
    np.set_printoptions(suppress=True)
    
    x = np.genfromtxt('Results/Optimisation_resultx21.csv', dtype=np.float64, delimiter=',')
    
    jac = Jacobian(x, -1, 0.001) # compile jit
    jac, base = jac_test(x, 0, 0.001)
    jac, base = jac_test(x, 1, 0.001)
    jac, base = jac_test(x, 2, 0.001)
    
    hes = hessian(x, 1)

 