# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

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

    zero = np.zeros(1, dtype=np.float64)

    for t in range(intervals):
        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1] if t>0 else 0.5 * Scapacity

        Discharget = np.minimum(np.minimum(np.maximum(zero, Netloadt), Pcapacity), Storaget_1 / resolution)
        Charget = np.minimum(np.minimum(-1 * np.minimum(zero, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency

        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget

    Deficit = np.maximum(Netload - Discharge, zero)
    Spillage = -1 * np.minimum(Netload + Charge, zero)

    solution.flexible = flexible
    solution.Spillage = Spillage
    solution.Charge = Charge
    solution.Discharge = Discharge
    solution.Storage = Storage
    solution.Deficit = Deficit

    return Deficit
