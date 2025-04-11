# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import datetime as dt

import numpy as np

from firm.Input import (
    Evaluate,
    Solution,
)
from firm.Simulation import Simulate
from firm.Utils import zero_safe_division

def Debug(solution):
    """Debugging"""

    for t in range(solution.intervals):

        # Energy supply-demand balance
        assert (
            np.abs(
                solution.MLoad[t]
                + solution.MCharge[t]
                + solution.MSpillage[t]
                - solution.MPV[t]
                - solution.MWind[t]
                - solution.CBaseload
                - solution.MFlexible[t]
                - solution.MDischarge[t]
                - solution.MDeficit[t]
                - (solution.TImport[t] + solution.TExport[t]).sum(axis=0)
            )
            < 0.001
        ).all(), f"Energy Balance, {t}"

        # Discharge, Charge and Storage
        if t == 0:
            assert (
                np.abs(
                    solution.MStorage[t]
                    - 0.5 * solution.CPHS
                    + solution.resolution * (solution.MDischarge[t] - solution.MCharge[t] * solution.efficiency)
                )
                <= 0.001
            ).all(), f"Phes behaviour, {t}"
        else:
            assert (
                np.abs(
                    solution.MStorage[t]
                    - solution.MStorage[t - 1]
                    + solution.resolution * (solution.MDischarge[t] - solution.MCharge[t] * solution.efficiency)
                )
                <= 0.001
            ).all(), f"Phes behaviour, {t}"

    try:
        assert solution.MPV.sum(axis=1).max() <= solution.CPV.sum()
        assert solution.MWind.sum(axis=1).max() <= solution.CWind.sum()
    
        assert ((solution.TImport.sum(axis=2) + solution.TExport.sum(axis=2)) < 0.001).all, "import/export imbalance (lines)"
        assert ((solution.TImport.sum(axis=1) + solution.TExport.sum(axis=1)) < 0.001).all, "import/export imbalance (nodes)"
    
        assert (solution.TImport.sum(axis=2).max(axis=0) - solution.CHVI <= 0.001).all(), "HVI bounds"
        assert (solution.TImport.min(axis=2).min(axis=0) >= -0.001).all(), "HVI bounds"
    
        assert (solution.TExport.min(axis=2).min(axis=0) + solution.CHVI >= -0.001).all(), "HVI bounds"
        assert (solution.TExport.max(axis=2).max(axis=0) <= 0.001).all(), "HVI bounds"
    
        assert (solution.MDischarge.max(axis=0) - solution.CPHP <= 0.001).all(), "Phes Discharge"
        assert (solution.MCharge.max(axis=0) - solution.CPHP <= 0.001).all(), "Phes Charge"
        assert (solution.MStorage.max(axis=0) - solution.CPHS <= 0.001).all(), "Phes SOC, too much"
        assert (solution.MStorage.min(axis=0) >= -0.001).all(), "Phes SOC, negative"
    except: 
        pass
    print("Debugging: everything is ok")

    return True


def LPGM(solution):
    """Load profiles and generation mix data"""

    Debug(solution)

    C = np.stack(
        (
            solution.MLoad.sum(axis=1),
            solution.MHydro.sum(axis=1),
            solution.MBio.sum(axis=1),
            solution.MPV.sum(axis=1),
            solution.MWind.sum(axis=1),
            solution.MDischarge.sum(axis=1),
            solution.MDeficit.sum(axis=1),
            -1 * solution.MSpillage.sum(axis=1),
            -1 * solution.MCharge.sum(axis=1),
            solution.MStorage.sum(axis=1),
        )
    )
    TDC = np.zeros((len(solution.network_mask), solution.intervals))
    TDC[solution.network_mask] = solution.TDC.T
    
    C = np.vstack(
        (
            C, 
            TDC
        )
    )
    C = np.around(1000.*C.T)

    datentime = np.array(
        [
            (dt.datetime(firstyear, 1, 1, 0, 0) + x * dt.timedelta(minutes=60 * solution.resolution)).strftime(
                "%a %d-%b %Y %H:%M"
            )
            for x in range(solution.intervals)
        ]
    )
    C = np.insert(C.astype("str"), 0, datentime, axis=1)

    header = (
        "Date & time,Operational demand,Hydropower,Biomass,Solar photovoltaics,Wind,"
        "Pumped hydro energy storage,Energy deficit,Energy spillage,PHES-Charge,"
        "PHES-Storage,FNQ-QLD,NSW-QLD,NSW-SA,NSW-VIC,NT-SA,SA-WA,TAS-VIC"
    )

    np.savetxt(f"../Results/S{solution.scenario}.csv", C, fmt="%s", delimiter=",", header=header, comments="")

    if solution.scenario >= 21:
        header = (
            "Date & time,Operational demand,Hydropower,Biomass,Solar photovoltaics,Wind,"
            "Pumped hydro energy storage,Energy deficit,Energy spillage,"
            "Transmission,PHES-Charge,PHES-Storage"
        )

        Nodel = np.array(["FNQ", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"])[solution.Nodel_int]

        for j, node in enumerate(Nodel):
            C = np.stack(
                [
                    solution.MLoad[:, j],
                    solution.MHydro[:, j],
                    solution.MBio[:, j],
                    solution.MPV[:, j],
                    solution.MWind[:, j],
                    solution.MDischarge[:, j],
                    solution.MDeficit[:, j],
                    -1 * solution.MSpillage[:, j],
                    solution.MImport[:, j],
                    -1 * solution.MCharge[:, j],
                    solution.MStorage[:, j],
                ]
            )
            C = np.around(1000.*C.T)

            C = np.insert(C.astype("str"), 0, datentime, axis=1)
            np.savetxt(
                f"../Results/S{solution.scenario}{node}.csv", C, fmt="%s", delimiter=",", header=header, comments=""
            )

    print("Load profiles and generation mix is produced.")

    return True


def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""


    GPV    = solution.MPV.sum()    * solution.resolution / solution.years
    GWind  = solution.MWind.sum()  * solution.resolution / solution.years
    GHydro = solution.MHydro.sum() * solution.resolution / solution.years
    GBio   = solution.MBio.sum()   * solution.resolution / solution.years
    GPHES  = solution.MDischarge.sum() * solution.resolution / solution.years 

    CFPV = 100*zero_safe_division(GPV, solution.CPV.sum() * 8760)
    CFWind = 100*zero_safe_division(GWind, solution.CWind.sum() * 8760)
    CFPHES = 100*zero_safe_division(GPHES, solution.CPHP.sum() * 8760 / 2 * (1+solution.efficiency)/2)

    LCOGP = zero_safe_division((cost_model.pv    * np.array([solution.CPV.sum(),    solution.CPV.sum(),    GPV])).sum(),    GPV*1000)
    LCOGW = zero_safe_division((cost_model.onsw  * np.array([solution.CWind.sum(),  solution.CWind.sum(),  GWind])).sum(),  GWind*1000)
    LCOGH = zero_safe_division((cost_model.hydro * np.array([solution.CHydro.sum(), solution.CHydro.sum(), GHydro])).sum(), GHydro*1000)
    LCOGB = zero_safe_division((cost_model.hydro * np.array([solution.CBio.sum(),   solution.CBio.sum(),   GBio])).sum(),   GBio*1000)
    LCOSP = zero_safe_division((cost_model.phes * np.array([solution.CPHP.sum(), solution.CPHS.sum(), solution.CPHP.sum(), GPHES, 1])).sum(), GPHES*1000)
    
    print("Levelised costs of electricity:")
    print("\u2022 LCOE:", solution.LCOE)
    print("\u2022 LCOG:", solution.LCOG)
    print("\u2022 LCOB:", solution.LCOB)
    print(f"\u2022 LCOG-PV: {LCOGP},  (CF: {CFPV}%)")
    print(f"\u2022 LCOG-Wind: {LCOGW}, (CF: {CFWind}%)")
    print("\u2022 LCOG-Hydro:", LCOGH)
    print("\u2022 LCOG-Bio:", LCOGB)
    print("\u2022 LCOB-Storage:", solution.LCOBS)
    print("\u2022 LCOB-Transmission:", solution.LCOBT)
    print("\u2022 LCOB-Spillage & loss:", solution.LCOBL)
    print("\u2022 CAPEX:", solution.CAPEX)
    print("\u2022 OPEX:", solution.OPEX)

    D = np.atleast_2d(np.array(
        [solution.energy / 1000_000, solution.CPV.sum(), GPV/1000, solution.CWind.sum(), GWind/1000, 
         solution.CHydro.sum() + solution.CBio.sum(), (GHydro+GBio)/1000, solution.CPHP.sum(), 
         solution.CPHS.sum(), GPHES/1000]
        + list(solution.CHVI)
        + [solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL, 
           LCOGP, LCOGW, solution.CAPEX, solution.OPEX]))
    
    Nodel = np.array(["FNQ", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"])[solution.Nodel_int]
    
    header = ','.join(['Energy (PWh p.a.)', 'Utility PV (GW)', 'Utility PV (TWh p.a.)', 
                       'Onshore Wind (GW)', 'Onshore Wind (TWh p.a.)', 'Hydro & Bio (GW)', 'Hydro & Bio (TWh p.a.)', 
                       'PHES capacity (GW)', 'PHES capacity (GWh)', 'PHES (TWh p.a.)'] +
                      [f'{Nodel[n[0]]}-{Nodel[n[1]]} (GW)' for n in solution.basic_network] +
                      ['LCOE', 'LCOG', 'LCOB (storage)', 'LCOB (transmission)', 'LCOB (curtailment)', 
                       'LCOE-PV', 'LCOE-Wind', 'CAPEX', 'OPEX']
                      )

    np.savetxt(f'../Results/GGTA{solution.scenario}.csv', D, fmt='%f', delimiter=',', header=header, comments='')
    print('Energy generation, storage and transmission information is produced.')

    return True


def Information(x):
    """Dispatch: Statistics.Information(x, Flex)"""

    start = dt.datetime.now()
    print("Statistics start at", start)

    S = Solution(x, -1, False)
    Evaluate(S, cost_model)
    try:
        assert S.MDeficit.sum() < 0.1, "Energy generation and demand are not balanced."
    except AssertionError:
        pass
    S.TDC = (np.atleast_3d(S.trans_mask).T * (S.TImport + S.TExport)).sum(axis=2)
    S.MImport = (S.TImport+S.TExport).sum(axis=1)

    S.MHydro = np.minimum(S.MFlexible, S.CHydro)
    S.MBio = np.minimum(S.MFlexible - S.MHydro, S.CBio)
    S.MHydro += S.CBaseload
    
    LPGM(S)
    GGTA(S)

    end = dt.datetime.now()
    print("Statistics took", end - start)

    return True


if __name__ == "__main__":
    Information(x0)
