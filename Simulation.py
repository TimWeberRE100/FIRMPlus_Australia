# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import jit

@jit(nopython=True)
def Reliability(solution, flexible):

    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) - solution.GBaseload.sum(axis=1)) - flexible # Sj-ENLoad(j, t)

    length = len(Netload)

    Pcapacity = solution.CPHP.sum() * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh

    Discharge = np.zeros(length)
    Charge = np.zeros(length)
    Storage = np.zeros(length)

    for t in range(length):

        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1] if t>0 else 0.5 * Scapacity

        Discharget = min(max(0, Netloadt), Pcapacity, Storaget_1 / solution.resolution)
        Charget = min(-1 * min(0, Netloadt), Pcapacity, (Scapacity - Storaget_1) / solution.efficiency / solution.resolution)
        Storaget = Storaget_1 - Discharget * solution.resolution + Charget * solution.resolution * solution.efficiency

        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget

    Deficit = np.maximum(Netload - Discharge, 0)
    Spillage = -1 * np.minimum(Netload + Charge, 0)

    solution.flexible = np.atleast_2d(flexible)
    solution.Spillage = np.atleast_2d(Spillage)
    solution.Charge = np.atleast_2d(Charge)
    solution.Discharge = np.atleast_2d(Discharge)
    solution.Storage = np.atleast_2d(Storage)
    solution.Deficit = np.atleast_2d(Deficit)

    return Deficit
