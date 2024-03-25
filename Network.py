# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import jit, int64

@jit(nopython=True)
def Transmission(solution, output=False):
    
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

    # if output:

    #     MStorage = np.tile(solution.Storage, (nodes, 1)).transpose() * pcfactor # SPH(t, j), MWh
    #     MDischargeD = np.tile(solution.DischargeD, (nodes, 1)).transpose() * pcfactorD  # MDischargeD: DD(j, t)
    #     MStorageD = np.tile(solution.StorageD, (nodes, 1)).transpose() * pcfactorD  # SD(t, j), MWh
    #     solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak = (MPV, MWind,MBaseload, MPeak)
    #     solution.MDischarge, solution.MCharge, solution.MStorage, solution.MP2V = (MDischarge, MCharge, MStorage, MP2V)
    #     solution.MDischargeD, solution.MChargeD, solution.MStorageD = (MDischargeD, MChargeD, MStorageD)
    #     solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)

    return TDC