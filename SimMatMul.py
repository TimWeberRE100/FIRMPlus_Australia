# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:03:33 2024

@author: u6942852
"""

import numpy as np
from numba import njit
    
perfect = np.array([0,1,3,6,10,15,21]) #that's more than enough for now
    
    
# @njit
def Reliability(solution, flexible, start=None, end=None):
    """ flexible = np.ones((intervals, nodes))*CPeak*1000 """
    network, directconns, nodeline_mask = solution.network, solution.directconns,solution.nodeline_mask

    networksteps = np.where(perfect == network.shape[1])[0][0]
    
    Netload = (solution.MLoad - solution.GPV - solution.GWind - solution.GBaseload)[start:end]
    Netload -= flexible
    
    shape2d = intervals, nodes = len(Netload), solution.nodes

    Pcapacity = solution.CPHP * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    Hcapacity = solution.CHVDC * 1000
    nhvdc = len(solution.CHVDC)
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(shape2d, dtype=np.float64)
    Charge = np.zeros(shape2d, dtype=np.float64)
    Storage = np.zeros(shape2d, dtype=np.float64)
    Deficit = np.zeros(shape2d, dtype=np.float64)
    Transmission = np.zeros(shape2d, dtype=np.float64)
    TDC = np.zeros((intervals, nhvdc))

    
    # hvdc_all = directconns[np.tril_indices(nodes, -1)]
    # hvmask = hvdc_all > -1
    # chvdc_all = np.zeros(hvdc_all.shape, dtype=np.float64)
    # chvdc_all[hvmask] = Hcapacity[hvdc_all[hvmask]]    

# =============================================================================
    nl_inds = np.array([[i,j] for i, j in zip(*np.where(nodeline_mask == True))])
    nli = np.empty((int(len(nl_inds)/2), 3), np.int64)
    for i, j in enumerate(np.unique(nl_inds[:,0])):
        nli[i] = np.concatenate((np.array([j]), nl_inds[nl_inds[:,0]==j, 1]))
# np.hstack((np.arange(len(networks[0])).reshape(-1, 1), networks[0]))
# =============================================================================
        

    for t in range(intervals):
        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1,:] if t>0 else 0.5*Scapacity
    
        Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        Deficitt = np.maximum(Netloadt - Discharget ,0)
    
        Transmissiont = np.zeros((nhvdc, nodes), dtype=np.float64)
        
        if Deficitt.sum() > 1e-6: 
            # Fill deficits with transmission by drawing down from neighbours battery reserves
            Fillt = np.maximum(Netloadt - Discharget, 0)
            Surplust = -1 * np.minimum(0, Netloadt + Charget) + np.minimum(Pcapacity, Storaget_1 / resolution)
            
            Transmissiont = hvdc(Fillt, Surplust, Transmissiont, Hcapacity, network, nodes, nli)
            
            Netloadt = Netload[t] - Transmissiont.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
    
        # =============================================================================
        # TODO: If deficit Go back in time and discharge batteries 
        # This may make the time a fair bit longer
        # =============================================================================
        
        Surplust = -1 * np.minimum(0, Netloadt + Charget) 
        if Surplust.sum() > 1e-6:
            # Distribute surplus energy with transmission to areas with spare charging capacity
            Fillt = (np.maximum(0, Netloadt) #load
                        + np.minimum(Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution) #full charging capacity
                        - Charget) #charge capacity already in use
    
            Transmissiont = hvdc(
                Fillt, Surplust, Transmissiont, Hcapacity, network, nodes, nli)
            
            Netloadt = Netload[t] - Transmissiont.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
    
        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        
        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget
        Transmission[t] = Transmissiont.sum(axis=0)
        TDC[t] = np.maximum(0, Transmissiont).sum(axis=1)
        
        
    Deficit = np.maximum(Netload - Transmission - Discharge, np.zeros_like(Netload))
    Spillage = -1 * np.minimum(Netload + Charge, np.zeros_like(Netload))

    solution.flexible = flexible
    solution.Spillage = Spillage
    solution.Charge = Charge
    solution.Discharge = Discharge
    solution.Storage = Storage
    solution.Deficit = Deficit
    solution.Transmission = Transmission
    solution.TDC = TDC
    
    
    return Deficit

def hvdc(Fillt, Surplust, Transmissiont, Hcapacity, network, nodes, nli):
    
    for n in range(len(Fillt)):
        if Fillt[n] == 0:
            continue
        
        ll = nli[(nli[:,1:]==n).sum(axis=1).astype(np.bool_), :]
        ll, ln = ll[:,0], ll[:,1:]
        ln = ln[ln != n]
        
        _trans = np.minimum(Surplust[ln], #surplus and line hosting capacity
                            Hcapacity[ll] - Transmissiont[ll].sum(axis=1))
        oversupply = _trans.sum()/Fillt[n]
        
        if oversupply > 1:
            _trans /= oversupply
        
        Fillt[n] -= _trans.sum()
        for i, l, d in enumerate(zip(ll, ln)):
            Surplust[d] -= _trans[i]
            Transmissiont[l, n] += _trans[i]
            Transmissiont[l, d] -= _trans[i]
        
    return Transmissiont

