# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit, jit
from numba import cuda, guvectorize, vectorize

@njit()
def Reliability(solution, flexible, start=None, end=None):
    """Single-solution version of Reliability"""
    assert solution.nvec == 1 
    assert solution.vectorised is False

    if start is None and end is None: 
        Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) -
                   solution.GBaseload.sum(axis=1) - flexible)
        intervals = solution.intervals

    else: 
        Netload = ((solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) -
                   solution.GBaseload.sum(axis=1))[start:end] - flexible)
        intervals = len(Netload)

    Pcapacity = solution.CPHP.sum() * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(intervals)
    Charge = np.zeros(intervals)
    Storage = np.zeros(intervals)

    for t in range(intervals):
        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1] if t>0 else 0.5*Scapacity

        Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        
        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget

    Deficit = np.maximum(Netload - Discharge, np.zeros(intervals))
    Spillage = -1 * np.minimum(Netload + Charge, np.zeros(intervals))

    solution.flexible = flexible
    solution.Spillage = Spillage
    solution.Charge = Charge
    solution.Discharge = Discharge
    solution.Storage = Storage
    solution.Deficit = Deficit

    return Deficit


@guvectorize(['void(float64[:], float64, float64, float64, float64, float64[:,:], float64[:,:])'],
        '(m),(),(),(),(),(m,n)->(m,n)',
           target='cuda'
           )
def _simulate(Netload, Pcapacity, Scapacity, resolution, efficiency, shape, Charging):
    Charging[-1, 2] = 0.5 * Scapacity
    for t in range(len(Netload)):
        Storaget_1 = Charging[t-1, 2]

        Charging[t, 0] = np.minimum(np.minimum(np.maximum(0.0, Netload[t]), Pcapacity), Storaget_1 / resolution)
        Charging[t, 1] = np.minimum(np.minimum(-1 * np.minimum(0.0, Netload[t]), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Charging[t, 2] = Storaget_1 - Charging[t, 0] * resolution + Charging[t, 1] * resolution * efficiency

"""


https://numba.discourse.group/t/using-guvectorize-inside-a-jitted-function/1966/9 


"""

@njit()
def VReliability(solution, flexible):
    """Vectorised version of Reliability"""
    shape2d = solution.intervals, solution.nvec
    intervals, nvec = shape2d
    
    assert solution.vectorised is True

    # Flexible must be 2D of shape (intervals, N) where N is broadcastable to nvec
    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) -
               solution.GBaseload.sum(axis=1) - flexible)

    Pcapacity = solution.CPHP.sum(axis=0) * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(shape2d)
    Charge = np.zeros(shape2d)
    Storage = np.zeros(shape2d)

    Charging = np.empty((intervals, nvec, 3), np.float64)
    _charging = np.empty((intervals, nvec, 3), np.float64)

    Charging = _simulate(Netload, Pcapacity, Scapacity, resolution, efficiency, _charging, Charging
    )
    
    Discharge, Charge, Storage = Charging[:, :, 0], Charging[:, :, 1], Charging[:, :, 2]

    Deficit = np.maximum(Netload - Discharge, 0.0)
    Spillage = -1 * np.minimum(Netload + Charge, 0.0)

    solution.flexible = flexible
    solution.Spillage = Spillage
    solution.Charge = Charge
    solution.Discharge = Discharge
    solution.Storage = Storage
    solution.Deficit = Deficit

    return Deficit
