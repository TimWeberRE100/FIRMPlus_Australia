# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:22 2024

@author: u6942852
"""

import numpy as np
from numba import njit  # type: ignore

from firm.Interconnection import Interconnection
from firm.Utils import cclock


@njit
def Simulate(solution):

    TransmissionSimulate(solution)

    # meet deficits in place 
    solution.MFlexible = np.minimum(solution.MDeficit, solution.CPeak)
    UpdateUnbalanced(solution)
    UpdateSpillDef(solution)
    if solution.profiling:
        time_adj = (solution.time_interconnection0+
                    solution.time_interconnection1+
                    solution.time_interconnection2+
                    solution.time_interconnection3)
        start_backfill = cclock()
    fill = np.zeros(solution.nodes, np.float64)
    for t in range(solution.intervals - 1, -1, -1):
        # timestep backwards
        if solution.MDeficit[t].sum() > 1e-6:
            # meet deficits just-in-time with flex
            # solution.MFlexible[t] = np.minimum(solution.MDeficit[t], solution.CPeak)
            # Deficit modified in place by Interconnection below, so emulate that behaviour here
            # solution.MDeficit[t] -= solution.MFlexible[t]
            # if remaining deficits:
            if solution.MDeficit[t].sum() > 1e-6:
                # original import/export
                _import, _export = solution.TImport[t].copy(), solution.TExport[t].copy()
                # meet deficits just-in-time by importiing flex from neighbours
                Interconnection(
                    solution,
                    solution.MDeficit[t],
                    solution.CPeak - solution.MFlexible[t],
                    solution.TImport[t],
                    solution.TExport[t],
                )
                # flexible += iexports from neighbours
                solution.MFlexible[t] += np.maximum(
                    0, (_import + _export - solution.TImport[t] - solution.TExport[t]).sum(axis=0)
                )
            # accumulate remaing deficits
            fill += solution.MDeficit[t] / solution.efficiency
        if fill.sum() > 1e-6:
            # clip fill by storage capacity
            fill = np.minimum(
                fill, (solution.CPHS - solution.MStorage[t - 1]) / solution.resolution / solution.efficiency
            )
            flex = np.minimum(
                np.minimum(fill, solution.CPeak - solution.MFlexible[t]),
                solution.CPHP - solution.MCharge[t] + solution.MDischarge[t],
            )
            fill -= flex
            solution.MFlexible[t] += flex
            if fill.sum() - flex.sum() > 1e-6:
                _import, _export = solution.TImport[t].copy(), solution.TExport[t].copy()
                Interconnection(
                    solution, fill, solution.CPeak - solution.MFlexible[t], solution.TImport[t], solution.TExport[t]
                )
                solution.MFlexible[t] += np.maximum(
                    0, (_import + _export - solution.TImport[t] - solution.TExport[t]).sum(axis=0)
                )
                # fill adjusted in-place
    if solution.profiling:
        time_adj -= (solution.time_interconnection0+
                     solution.time_interconnection1+
                     solution.time_interconnection2+
                     solution.time_interconnection3) 
        solution.time_backfill += cclock() - start_backfill + time_adj
        solution.calls_backfill +=1 

    BasicSimulate(solution)

@njit
def TransmissionSimulate(solution):
    if solution.profiling:
        start_transmission = cclock()
        time_adj = (solution.time_interconnection0+
                    solution.time_interconnection1+
                    solution.time_interconnection2+
                    solution.time_interconnection3+
                    solution.time_storage_behaviort+
                    solution.time_spilldeft+
                    solution.time_unbalancedt+
                    solution.time_update_soct)
    for t in range(solution.intervals):
        # storage operation
        UpdateStoraget(solution, t)
        UpdateSpillDeft(solution, t)

        # fill deficits from spilled power
        if solution.MDeficit[t].sum() > 1e-6:
            if solution.MSpillage[t].sum() > 1e-6:
                Interconnection(
                    solution, solution.MDeficit[t], solution.MSpillage[t], solution.TImport[t], solution.TExport[t]
                )
                # update storage behaviour
                UpdateUnbalancedt(solution, t)
                UpdateStoraget(solution, t)
                # UpdateSpillDeft(solution, t) # updated inplace by Interconnection

        # fill deficits by drawing down neighbours' storage reserves
        if solution.MDeficit[t].sum() > 1e-6:
            Surplust = np.maximum(
                0,
                solution.MSpillage[t]
                + solution.MCharge[t]
                + np.minimum(solution.CPHP, solution.MStorage[t - 1] / solution.resolution)
                - solution.MDischarge[t],
            )
            if Surplust.sum() > 1e-6:
                Interconnection(solution, solution.MDeficit[t], Surplust, solution.TImport[t], solution.TExport[t])
                # update storage behaviour
                UpdateUnbalancedt(solution, t)
                UpdateStoraget(solution, t)
                UpdateSpillDeft(solution, t)

        UpdateSOCt(solution, t)

    if solution.MDeficit.sum() < 1e-6:
        if solution.profiling:
            time_adj -= (solution.time_interconnection0+
                    solution.time_interconnection1+
                    solution.time_interconnection2+
                    solution.time_interconnection3+
                    solution.time_storage_behaviort+
                    solution.time_spilldeft+
                    solution.time_unbalancedt+
                    solution.time_update_soct)
            solution.time_transmission += cclock() - start_transmission + time_adj
            solution.calls_transmission +=1
        return

    # precharge batteries with spillage only 
    fill = np.zeros(solution.nodes, dtype=np.float64)
    # timestep backwards
    for t in range(solution.intervals - 1, -1, -1):
        if fill.sum() > 1e-6 and solution.MSpillage[t].sum() > 1e-6:
            # cap fill by storage capacity
            fill = np.minimum(
                fill, (solution.CPHS - solution.MStorage[t - 1]) / solution.resolution / solution.efficiency
            )
            # meet fill with neighbours' spillage - don't draw down power as this affects future SOC
            Interconnection(solution, fill, solution.MSpillage[t], solution.TImport[t], solution.TExport[t])
            # fill adjusted in-place
        fill += solution.MDeficit[t] / solution.efficiency

    if solution.profiling:
        time_adj -= (solution.time_interconnection0+
                    solution.time_interconnection1+
                    solution.time_interconnection2+
                    solution.time_interconnection3+
                    solution.time_storage_behaviort+
                    solution.time_spilldeft+
                    solution.time_unbalancedt+
                    solution.time_update_soct)
        solution.time_transmission += cclock() - start_transmission + time_adj
        solution.calls_transmission +=1 

    # fix storage traces
    BasicSimulate(solution)


@njit
def BasicSimulate(solution):
    # if solution.profiling:
    #     start_basic = cclock()
    solution.MStorage[-1] = 0.5 * solution.CPHS 
    UpdateUnbalanced(solution)
    UpdateStorage(solution)
    UpdateSOC(solution)
    # UpdateSpillDef(solution)
    # if solution.profiling:
    #     solution.time_basic += cclock() - start_basic
    #     solution.calls_basic +=1 


@njit
def UpdateUnbalancedt(solution, t):
    if solution.profiling:
        start = cclock()
    solution.MUnbalanced[t] = solution.MNetload[t] - solution.MFlexible[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0)
    if solution.profiling:
        solution.time_unbalancedt += cclock() - start
        solution.calls_unbalancedt +=1 


@njit
def UpdateUnbalanced(solution):
    if solution.profiling:
        start = cclock()
    solution.MUnbalanced = solution.MNetload - solution.MFlexible - (solution.TImport + solution.TExport).sum(axis=1)
    if solution.profiling:
        solution.time_unbalanced += cclock() - start
        solution.calls_unbalanced +=1 


@njit
def UpdateStoraget(solution, t):
    if solution.profiling:
        start = cclock()
    solution.MCharge[t] = np.minimum(
        np.minimum(
            -np.minimum(0, solution.MUnbalanced[t]), # available/required power
            solution.CPHP, # storage power constraint
        ),  
        (solution.CPHS - solution.MStorage[t - 1]) / solution.efficiency / solution.resolution, # storage energy constraint
    )  
    solution.MDischarge[t] = np.minimum(
        np.minimum(
            np.maximum(0, solution.MUnbalanced[t]),# available/required power
            solution.CPHP,# storage power constraint
        ),  
        solution.MStorage[t - 1] / solution.resolution, # storage energy constraint
    )  
    if solution.profiling:
        solution.time_storage_behaviort += cclock() - start
        solution.calls_storage_behaviort +=1 


@njit
def UpdateStorage(solution):
    if solution.profiling:
        start = cclock()
    solution.MCharge = np.minimum(
        -np.minimum(0, solution.MUnbalanced),
        solution.CPHP)
    solution.MDischarge = np.minimum(
        np.maximum(0, solution.MUnbalanced),
        solution.CPHP)
    if solution.profiling:
        solution.time_storage_behavior += cclock() - start
        solution.calls_storage_behavior +=1 

    
@njit
def UpdateSOCt(solution, t):
    if solution.profiling:
       start = cclock()
    solution.MStorage[t] = solution.MStorage[t - 1] + solution.resolution * (
        solution.MCharge[t] * solution.efficiency - solution.MDischarge[t])
    if solution.profiling:
        solution.time_update_soct += cclock() - start
        solution.calls_update_soct +=1 

@njit 
def UpdateSOC(solution):
    if solution.profiling:
        start = cclock()

    for t in range(solution.intervals):
        solution.MCharge[t] = np.minimum(
            solution.MCharge[t], 
            (solution.CPHS - solution.MStorage[t - 1]) / solution.efficiency / solution.resolution
            )
        solution.MDischarge[t] = np.minimum(
            solution.MDischarge[t], 
            solution.MStorage[t - 1] / solution.resolution,
            )
        solution.MStorage[t] = solution.MStorage[t - 1] + solution.resolution * (
            solution.MCharge[t] * solution.efficiency - solution.MDischarge[t]
        )
    if solution.profiling:
        solution.time_update_soc += cclock() - start
        solution.calls_update_soc +=1 

@njit
def UpdateSpillDeft(solution, t):
    if solution.profiling:
        start = cclock()
    solution.MDeficit[t] = np.maximum(
        0,
        solution.MUnbalanced[t]
        + solution.MCharge[t]
        - solution.MDischarge[t],
    )
    solution.MSpillage[t] = -np.minimum(
        0,
        solution.MUnbalanced[t]
        + solution.MCharge[t]
        - solution.MDischarge[t],
    )
    if solution.profiling:
        solution.time_spilldeft += cclock() - start
        solution.calls_spilldeft +=1 

@njit
def UpdateSpillDef(solution):
    if solution.profiling:
        start = cclock()

    solution.MDeficit = np.maximum(
        0,
        solution.MUnbalanced
        + solution.MCharge
        - solution.MDischarge,
    )
    solution.MSpillage = -np.minimum(
        0,
        solution.MUnbalanced
        + solution.MCharge
        - solution.MDischarge,
    )
    if solution.profiling:
        solution.time_spilldef += cclock() - start
        solution.calls_spilldef +=1 
