# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import jit

@jit(nopython=True, parallel=True)
def Reliability(solution, flexible):
    shape2d = solution.intervals, solution.nvec
    length, nvec = shape2d

    # Flexible must be 2D of shape (intervals, N) where N is broadcastable to nvec

    # jit with parallel=True does not support automatic broadcasting
    Netload = (np.broadcast_to(solution.MLoad.sum(axis=1), shape2d) +
                np.broadcast_to(solution.GPV.sum(axis=1), shape2d) +
                np.broadcast_to(solution.GWind.sum(axis=1), shape2d) +
                np.broadcast_to(solution.GBaseload.sum(axis=1), shape2d) -
                np.broadcast_to(flexible, shape2d)
                )

    Pcapacity = solution.CPHP.sum(axis=0) * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(shape2d)
    Charge = np.zeros(shape2d)
    Storage = np.zeros(shape2d)

    zero = np.zeros(1, dtype=np.float64)

    for t in range(length):
        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1] if t>0 else 0.5 * Scapacity

        Discharget = np.minimum(np.minimum(np.maximum(zero, Netloadt), Pcapacity), Storaget_1 / resolution)
        Charget = np.minimum(np.minimum(-1 * np.minimum(zero, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Storaget = Storaget_1 + (Charget * efficiency - Discharget) * resolution

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
