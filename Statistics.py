# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from Network import Transmission

import numpy as np
import datetime as dt

def Debug(solution):
    """Debugging"""

    Load, PV, Wind = (solution.MLoad.sum(axis=1), solution.GPV.sum(axis=1), solution.GWind.sum(axis=1))
    Baseload, Peak = (solution.MBaseload.sum(axis=1), solution.MPeak.sum(axis=1))

    Discharge, Charge, Storage = (solution.Discharge, solution.Charge, solution.Storage)
    Deficit, Spillage = (solution.Deficit, solution.Spillage)

    PHS = solution.CPHS * pow(10, 3) # GWh to MWh
    efficiency = solution.efficiency

    for i in range(intervals):
        # Energy supply-demand balance
        assert abs(Load[i] + Charge[i] + Spillage[i]
                   - PV[i] - Wind[i] - Baseload[i] - Peak[i] - Discharge[i] - Deficit[i]) <= 1

        # Discharge, Charge and Storage
        if i==0:
            assert abs(Storage[i] - 0.5 * PHS + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
        else:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1

        # Capacity: PV, wind, Discharge, Charge and Storage
        try:
            assert np.amax(PV) <= sum(solution.CPV) * pow(10, 3), print(np.amax(PV) - sum(solution.CPV) * pow(10, 3))
            assert np.amax(Wind) <= sum(solution.CWind) * pow(10, 3), print(np.amax(Wind) - sum(solution.CWind) * pow(10, 3))

            assert np.amax(Discharge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Discharge) - sum(solution.CPHP) * pow(10, 3))
            assert np.amax(Charge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Charge) - sum(solution.CPHP) * pow(10, 3))
            assert np.amax(Storage) <= solution.CPHS * pow(10, 3), print(np.amax(Storage) - solution.CPHS * pow(10, 3))
        except AssertionError:
            pass

    print('Debugging: everything is ok')

    return True

def LPGM(solution):
    """Load profiles and generation mix data"""

    Debug(solution)

    C = np.stack([solution.MLoad.sum(axis=1), solution.MHydro.sum(axis=1), solution.MBio.sum(axis=1), 
                  solution.GPV.sum(axis=1), solution.GWind.sum(axis=1), solution.Discharge, 
                  solution.Deficit, -1 * solution.Spillage, -1 * solution.Charge, solution.Storage,
                  solution.FQ, solution.NQ, solution.NS, solution.NV, solution.AS, solution.SW, solution.TV])
    C = np.around(C.transpose())

    datentime = np.array([(dt.datetime(firstyear, 1, 1, 0, 0) + x * dt.timedelta(minutes=60 * resolution)).strftime('%a %d-%b %Y %H:%M') for x in range(intervals)])
    C = np.insert(C.astype('str'), 0, datentime, axis=1)

    header = 'Date & time,Operational demand,Hydropower,Biomass,Solar photovoltaics,Wind,' \
             'Pumped hydro energy storage,Energy deficit,Energy spillage,PHES-Charge,' \
             'PHES-Storage,FNQ-QLD,NSW-QLD,NSW-SA,NSW-VIC,NT-SA,SA-WA,TAS-VIC'

    np.savetxt('Results/S{}.csv'.format(scenario), C, fmt='%s', delimiter=',', header=header, comments='')

    if scenario>=21:
        header = 'Date & time,Operational demand,Hydropower,Biomass,Solar photovoltaics,Wind,' \
                 'Pumped hydro energy storage,Energy deficit,Energy spillage,' \
                 'Transmission,PHES-Charge,PHES-Storage'

        Topology = solution.Topology[np.where(np.in1d(np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']), coverage) == True)[0]]

        for j in range(nodes):
            C = np.stack([(solution.MLoad)[:, j], solution.MHydro[:, j], solution.MBio[:, j], solution.MPV[:, j], 
                          solution.MWind[:, j], solution.MDischarge[:, j], solution.MDeficit[:, j],
                          -1 * solution.MSpillage[:, j], Topology[j], -1 * solution.MCharge[:, j],
                          solution.MStorage[:, j]])
            C = np.around(C.transpose())

            C = np.insert(C.astype('str'), 0, datentime, axis=1)
            np.savetxt('Results/S{}{}.csv'.format(scenario, Nodel[j]), C, fmt='%s', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True

def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""

    factor = np.genfromtxt('Data/factor.csv', dtype=None, delimiter=',', encoding=None)
    factor = dict(factor)

    CPV, CWind, CPHP, CPHS = (sum(solution.CPV), sum(solution.CWind), sum(solution.CPHP), solution.CPHS) # GW, GWh
    CapHydro, CapBio = CHydro.sum(), CBio.sum() # GW
    CapHydrobio = CapHydro + CapBio

    GPV, GWind, GHydro, GBio = map(lambda x: x * pow(10, -6) * resolution / years, (solution.GPV.sum(), solution.GWind.sum(), solution.MHydro.sum(), solution.MBio.sum())) # TWh p.a.
    GHydrobio = GHydro + GBio
    CFPV, CFWind = (GPV / CPV / 8.76, GWind / CWind / 8.76)

    CostPV = factor['PV'] * CPV # A$b p.a.
    CostWind = factor['Wind'] * CWind # A$b p.a.
    CostHydro = factor['Hydro'] * GHydro # A$b p.a.
    CostBio = factor['Hydro'] * GBio # A$b p.a.
    CostPH = factor['PHP'] * CPHP + factor['PHS'] * CPHS # A$b p.a.
    if scenario>=21:
        CostPH -= factor['LegPH']

    CostDC = np.array([factor['FQ'], factor['NQ'], factor['NS'], factor['NV'], factor['AS'], factor['SW'], factor['TV']])
    CostDC = (CostDC * solution.CDC).sum() # A$b p.a.
    if scenario>=21:
        CostDC -= factor['LegINTC']

    CostAC = factor['ACPV'] * CPV + factor['ACWind'] * CWind # A$b p.a.

    Energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
    Loss = np.sum(abs(solution.TDC), axis=0) * DCloss
    Loss = Loss.sum() * pow(10, -9) * resolution / years # PWh p.a.

    LCOE = (CostPV + CostWind + CostHydro + CostBio + CostPH + CostDC + CostAC) / (Energy - Loss)
    LCOG = (CostPV + CostWind + CostHydro + CostBio) * pow(10, 3) / (GPV + GWind + GHydro + GBio)
    LCOGP = CostPV * pow(10, 3) / GPV if GPV!=0 else 0
    LCOGW = CostWind * pow(10, 3) / GWind if GWind!=0 else 0
    LCOGH = CostHydro * pow(10, 3) / GHydro if GHydro!=0 else 0
    LCOGB = CostBio * pow(10, 3) / GBio if GBio!=0 else 0

    LCOB = LCOE - LCOG
    LCOBS = CostPH / (Energy - Loss)
    LCOBT = (CostDC + CostAC) / (Energy - Loss)
    LCOBL = LCOB - LCOBS - LCOBT

    print('Levelised costs of electricity:')
    print('\u2022 LCOE:', LCOE)
    print('\u2022 LCOG:', LCOG)
    print('\u2022 LCOB:', LCOB)
    print('\u2022 LCOG-PV:', LCOGP, '(%s)' % CFPV)
    print('\u2022 LCOG-Wind:', LCOGW, '(%s)' % CFWind)
    print('\u2022 LCOG-Hydro:', LCOGH)
    print('\u2022 LCOG-Bio:', LCOGB)
    print('\u2022 LCOB-Storage:', LCOBS)
    print('\u2022 LCOB-Transmission:', LCOBT)
    print('\u2022 LCOB-Spillage & loss:', LCOBL)

    D = np.zeros((1, 22))
    D[0, :] = [Energy * pow(10, 3), Loss * pow(10, 3), CPV, GPV, CWind, GWind, CapHydrobio, GHydrobio, CPHP, CPHS] \
              + list(solution.CDC) \
              + [LCOE, LCOG, LCOBS, LCOBT, LCOBL]

    np.savetxt('Results/GGTA{}.csv'.format(scenario), D, fmt='%f', delimiter=',')
    print('Energy generation, storage and transmission information is produced.')

    return True

def Information(x, flexible):
    """Dispatch: Statistics.Information(x, Flex)"""

    start = dt.datetime.now()
    print("Statistics start at", start)

    S = Solution(x)
    Deficit = Reliability(S, flexible=flexible)

    try:
        assert Deficit.sum() * resolution < 0.1, 'Energy generation and demand are not balanced.'
    except AssertionError:
        pass

    if scenario>=21:
        S.TDC = Transmission(S, output=True) # TDC(t, k), MW
    else:
        S.TDC = np.zeros((intervals, len(DCloss))) # TDC(t, k), MW

        S.MPeak = np.tile(flexible, (nodes, 1)).transpose() # MW
        S.MBaseload = GBaseload.copy() # MW

        S.MPV = S.GPV.sum(axis=1) if S.GPV.shape[1]>0 else np.zeros((intervals, 1))
        S.MWind = S.GWind.sum(axis=1) if S.GWind.shape[1]>0 else np.zeros((intervals, 1))

        S.MDischarge = np.tile(S.Discharge, (nodes, 1)).transpose()
        S.MDeficit = np.tile(S.Deficit, (nodes, 1)).transpose()
        S.MCharge = np.tile(S.Charge, (nodes, 1)).transpose()
        S.MStorage = np.tile(S.Storage, (nodes, 1)).transpose()
        S.MSpillage = np.tile(S.Spillage, (nodes, 1)).transpose()

    S.CDC = np.amax(abs(S.TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    S.FQ, S.NQ, S.NS, S.NV, S.AS, S.SW, S.TV = map(lambda k: S.TDC[:, k], range(S.TDC.shape[1]))

    S.MHydro = np.tile(CHydro - CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW
    S.MHydro = np.minimum(S.MHydro, S.MPeak)
    S.MBio = S.MPeak - S.MHydro
    S.MHydro += S.MBaseload

    S.Topology = np.array([-1 * S.FQ, -1 * (S.NQ + S.NS + S.NV), -1 * S.AS, S.FQ + S.NQ, S.NS + S.AS - S.SW, -1 * S.TV, S.NV + S.TV, S.SW])

    LPGM(S)
    GGTA(S)

    end = dt.datetime.now()
    print("Statistics took", end - start)

    return True

if __name__ == '__main__':
    capacities = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',')
    flexible = np.ones((intervals, ), dtype=np.float64)*CPeak.sum()*1000
    
    # flexible = np.genfromtxt('Results/Dispatch_Flexible{}.csv'.format(scenario), delimiter=',', skip_header=1)
    # capacities=np.array([0.78125,0.78125,0.78125,0.78125,0.78125,2.34375,0.78125,
    #                      0.78125,2.34375,0.78125,2.34375,0.78125,2.34375,2.34375,
    #                      2.34375,0.78125,0.78125,0.78125,1.5625,1.5625,1.5625,1.5625,
    #                      1.5625,4.6875,1.5625,4.6875,1.5625,1.5625,1.5625,1.5625,1.5625,
    #                      1.5625,1.5625,1.5625,1.5625,1.5625,1.5625,1.5625,4.6875,1.5625,
    #                      1.5625,1.5625,4.999287150973348,3.9837339973767265,
    #                      2.3005982340601734,1.9956554787796343,6.920525089860547,218.75])
    S=Solution(capacities)
    Reliability(S, flexible)
    flexible=np.maximum(0, S.flexible-S.Spillage)
    
    # Information(capacities, flexible)
    Information(capacities, np.zeros(intervals))
    
    
    