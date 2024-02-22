# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import jit

@jit(nopython=True)
def Transmission(solution, output=False):
    MPV = np.zeros((solution.nodes, solution.intervals))
    MWind = np.zeros((solution.nodes, solution.intervals))

    for i, j in enumerate(solution.Nodel_int):
        MPV[i, :] = solution.GPV[:, np.where(solution.PVl_int==j)[0]].sum(axis=1)
        MWind[i, :] = solution.GWind[:, np.where(solution.Windl_int==j)[0]].sum(axis=1)
    MPV, MWind = (MPV.transpose(), MWind.transpose()) # Sij-GPV(t, i), Sij-GWind(t, i), MW

    MBaseload = solution.GBaseload # MW
    pkfactor = solution.CPeak / solution.CPeak.sum()
    MPeak = solution.flexible.transpose() * np.atleast_2d(pkfactor)

    MLoad_denominator = np.atleast_2d(solution.MLoad.sum(axis=1))
    defactor = np.divide(solution.MLoad, MLoad_denominator.transpose())
    MDeficit = solution.Deficit.transpose() * defactor # MDeficit: EDE(j, t)

    MPW = MPV + MWind
    MPW_denominator = np.atleast_2d(MPW.sum(axis=1) + 0.000001)
    spfactor = np.divide(MPW, MPW_denominator.transpose())
    MSpillage = solution.Spillage.transpose() * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    pcfactor = np.atleast_2d(CPHP / sum(CPHP) if sum(CPHP) != 0 else np.zeros_like(CPHP))
    MDischarge = solution.Discharge.transpose() * pcfactor # MDischarge: DPH(j, t)
    MCharge = solution.Charge.transpose() * pcfactor # MCharge: CHPH(j, t)

    MImport = solution.MLoad + MCharge + MSpillage \
              - MPV - MWind - MBaseload - MPeak - MDischarge - MDeficit # EIM(t, j), MW

    # ['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'] = [0, 1, 2, 3, 4, 5, 6, 7] for Nodel = Nodel_int

    FQ = -1 * MImport[:, np.where(solution.Nodel_int==0)[0][0]] if 0 in solution.Nodel_int else np.zeros(solution.intervals)
    AS = -1 * MImport[:, np.where(solution.Nodel_int==2)[0][0]] if 2 in solution.Nodel_int else np.zeros(solution.intervals)
    SW = MImport[:, np.where(solution.Nodel_int==7)[0][0]] if 7 in solution.Nodel_int else np.zeros(solution.intervals)
    TV = -1 * MImport[:, np.where(solution.Nodel_int==5)[0][0]]

    NQ = MImport[:, np.where(solution.Nodel_int==3)[0][0]] - FQ
    NV = MImport[:, np.where(solution.Nodel_int==6)[0][0]] - TV

    NS = -1 * MImport[:, np.where(solution.Nodel_int==1)[0][0]] - NQ - NV
    NS1 = MImport[:, np.where(solution.Nodel_int==4)[0][0]] - AS + SW
    #assert abs(NS - NS1).max()<=0.1, print(abs(NS - NS1).max())

    TDC = np.vstack((FQ, NQ, NS, NV, AS, SW, TV)).transpose() # TDC(t, k), MW

    """ if output:
        MStorage = np.tile(solution.Storage, (nodes, 1)).transpose() * pcfactor # SPH(t, j), MWh
        MDischargeD = np.tile(solution.DischargeD, (nodes, 1)).transpose() * pcfactorD  # MDischargeD: DD(j, t)
        MStorageD = np.tile(solution.StorageD, (nodes, 1)).transpose() * pcfactorD  # SD(t, j), MWh
        solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak = (MPV, MWind,MBaseload, MPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage, solution.MP2V = (MDischarge, MCharge, MStorage, MP2V)
        solution.MDischargeD, solution.MChargeD, solution.MStorageD = (MDischargeD, MChargeD, MStorageD)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage) """

    return TDC