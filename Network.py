# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit, int64

@njit()
def Transmission(solution, output=False):
    
    assert solution.vectorised is False
    nodes, intervals = solution.nodes, solution.intervals
    MPV, MWind = np.zeros((nodes, intervals)), np.zeros((nodes, intervals))
    
    PVl_int, Windl_int = solution.PVl_int, solution.Windl_int
        
    for i, j in enumerate(solution.Nodel_int):
        MPV[i, :] = solution.GPV[:, np.where(PVl_int==j)[0]].sum(axis=1)
        MWind[i, :] = solution.GWind[:, np.where(Windl_int==j)[0]].sum(axis=1)
    MPV, MWind = MPV.T, MWind.T # Sij-GPV(t, i), Sij-GWind(t, i), MW

    MBaseload = solution.GBaseload # MW
    pkfactor = solution.CPeak / solution.CPeak.sum()
    # Assume flexible.shape is (intervals, )
    flexible = solution.flexible.copy()
    MPeak = (flexible.reshape(-1,1) * pkfactor.reshape(1,-1))
        
    MLoad_denominator = solution.MLoad.sum(axis=1)
    defactor = np.divide(solution.MLoad, MLoad_denominator.reshape(-1, 1))
    
    MDeficit = solution.Deficit.copy() # avoids numba error with reshaping below (also MSpillage, MCharge, MDischarge)
    MDeficit = MDeficit.reshape(-1, 1)  * defactor # MDeficit: EDE(j, t)

    MPW = MPV + MWind
    MPW_denominator = np.atleast_2d(MPW.sum(axis=1) + 0.00000001).T
    spfactor = np.divide(MPW, MPW_denominator)
    MSpillage = solution.Spillage.copy()
    MSpillage = MSpillage.reshape(-1, 1) * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    
    dzsm = CPHP != 0 # divide by zero safe mask
    pcfactor = np.zeros(CPHP.shape)
    pcfactor[dzsm] =  CPHP[dzsm] / CPHP[dzsm].sum(axis=0)
    
    MCharge, MDischarge = solution.Charge.copy(), solution.Discharge.copy()
    MDischarge = (MDischarge.reshape(-1, 1) * pcfactor)# MDischarge: DPH(j, t)
    MCharge = (MCharge.reshape(-1, 1) * pcfactor) # MCharge: CHPH(j, t)

    MImport = solution.MLoad + MCharge + MSpillage \
              - MPV - MWind - MBaseload - MPeak - MDischarge - MDeficit # EIM(t, j), MW

    # ['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'] = [0, 1, 2, 3, 4, 5, 6, 7] for Nodel = Nodel_int

    FQ = -1 * MImport[:, np.where(solution.Nodel_int==0)[0][0]] if 0 in solution.Nodel_int else np.zeros(intervals, dtype=np.float64)
    AS = -1 * MImport[:, np.where(solution.Nodel_int==2)[0][0]] if 2 in solution.Nodel_int else np.zeros(intervals, dtype=np.float64)
    SW = MImport[:, np.where(solution.Nodel_int==7)[0][0]] if 7 in solution.Nodel_int else np.zeros(intervals, dtype=np.float64)
    TV = -1 * MImport[:, np.where(solution.Nodel_int==5)[0][0]]

    NQ = MImport[:, np.where(solution.Nodel_int==3)[0][0]] - FQ
    NV = MImport[:, np.where(solution.Nodel_int==6)[0][0]] - TV

    NS = -1. * MImport[:, np.where(solution.Nodel_int==1)[0][0]] - NQ - NV
    NS1 = MImport[:, np.where(solution.Nodel_int==4)[0][0]] - AS + SW
    #assert abs(NS - NS1).max()<=0.1, print(abs(NS - NS1).max())

    TDC = np.stack((FQ, NQ, NS, NV, AS, SW, TV), axis=1) # TDC(t, k), MW

    if output is True:
        MStorage = solution.Storage.copy()
        MStorage = MStorage.reshape(-1,1) * pcfactor # SPH(t, j), MWh
        solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak = (MPV, MWind,MBaseload, MPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage = (MDischarge, MCharge, MStorage)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)

    return TDC

@njit()
def VTransmission(solution, output=False):
    """Vectorised version of transmission"""
    assert solution.vectorised is True

    nodes, intervals, nvec = solution.nodes, solution.intervals, solution.nvec
    MPV, MWind = np.zeros((nodes, intervals, nvec)), np.zeros((nodes, intervals, nvec))
    
    PVl_int, Windl_int = solution.PVl_int, solution.Windl_int
        
    for i, j in enumerate(solution.Nodel_int):
        MPV[i, :, :] = solution.GPV[:, np.where(PVl_int==j)[0], :].sum(axis=1)
        MWind[i, :, :] = solution.GWind[:, np.where(Windl_int==j)[0], :].sum(axis=1)
    MPV, MWind = (MPV.transpose(1, 0, 2), MWind.transpose(1, 0, 2)) # Sij-GPV(t, i), Sij-GWind(t, i), MW

    MBaseload = solution.GBaseload # MW
    pkfactor = solution.CPeak / solution.CPeak.sum()
    # Assume flexible is (intervals, 1)
    MPeak = (solution.flexible * pkfactor).reshape(intervals, nodes, -1)
        
    MLoad_denominator = np.atleast_2d(solution.MLoad.sum(axis=1))
    defactor = np.divide(solution.MLoad, MLoad_denominator.reshape(-1, 1, 1))
    
    MDeficit = solution.Deficit.copy() # avoids numba error with reshaping below (also MSpillage, MCharge, MDischarge)
    MDeficit = MDeficit.reshape(intervals, int64(1), nvec)  * defactor # MDeficit: EDE(j, t)

    MPW = MPV + MWind
    MPW_denominator = np.atleast_3d(MPW.sum(axis=1) + 0.000001)
    spfactor = np.divide(MPW, MPW_denominator.transpose(0, 2, 1))
    MSpillage = solution.Spillage.copy()
    MSpillage = MSpillage.reshape(intervals, nvec, int64(1)).transpose(0, 2, 1) * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    
    dzsm = CPHP.sum(axis=0) != 0 # divide by zero safe mask
    pcfactor = np.zeros(CPHP.shape)
    pcfactor[:, dzsm] =  CPHP[:, dzsm] / CPHP[:, dzsm].sum(axis=0)
    
    MCharge, MDischarge = solution.Charge.copy(), solution.Discharge.copy()
    MDischarge = (MDischarge.reshape(intervals, nvec, int64(1)).transpose(0,2,1) * pcfactor)# MDischarge: DPH(j, t)
    MCharge = (MCharge.reshape(intervals, nvec, int64(1)).transpose(0,2,1) * pcfactor) # MCharge: CHPH(j, t)

    MImport = solution.MLoad + MCharge + MSpillage \
              - MPV - MWind - MBaseload - MPeak - MDischarge - MDeficit # EIM(t, j), MW

    # ['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'] = [0, 1, 2, 3, 4, 5, 6, 7] for Nodel = Nodel_int

    FQ = -1 * MImport[:, np.where(solution.Nodel_int==0)[0][0], :] if 0 in solution.Nodel_int else np.zeros((intervals, nvec), dtype=np.float64)
    AS = -1 * MImport[:, np.where(solution.Nodel_int==2)[0][0], :] if 2 in solution.Nodel_int else np.zeros((intervals, nvec), dtype=np.float64)
    SW = MImport[:, np.where(solution.Nodel_int==7)[0][0], :] if 7 in solution.Nodel_int else np.zeros((intervals, nvec), dtype=np.float64)
    TV = -1 * MImport[:, np.where(solution.Nodel_int==5)[0][0], :]

    NQ = MImport[:, np.where(solution.Nodel_int==3)[0][0], :] - FQ
    NV = MImport[:, np.where(solution.Nodel_int==6)[0][0], :] - TV

    NS = -1. * MImport[:, np.where(solution.Nodel_int==1)[0][0], :] - NQ - NV
    NS1 = MImport[:, np.where(solution.Nodel_int==4)[0][0], :] - AS + SW
    #assert abs(NS - NS1).max()<=0.1, print(abs(NS - NS1).max())

    TDC = np.stack((FQ, NQ, NS, NV, AS, SW, TV), axis=2).transpose(0,2,1) # TDC(t, k), MW

    if output is True:
        MStorage = solution.Storage.copy()
        MStorage = np.atleast_3d(MStorage).transpose(0,2,1) * np.atleast_3d(pcfactor).transpose(2,0,1) # SPH(t, j), MWh
        solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak = (MPV, MWind,MBaseload, MPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage = (MDischarge, MCharge, MStorage)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)

    return TDC