# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from warnings import warn

class BehaviourWarning(Warning):
    def __init__(self, message):
        self.message=message
    def __str__(self):
        return repr(self.message)

def Debug(solution):
    """Debugging"""

    for t in range(solution.intervals):
        # supply-demand
        assert (np.abs(solution.Load[t] + solution.Spillage[t] + solution.Charge[t]
                       - solution.Discharge[t] - solution.Hydro[t] - solution.Bio[t]
                       + solution.Transmission[t] - solution.PV[t] - solution.Wind[t]) <= 1#MW
                ).all(), f"Supply demand unbalanced. t: {t}"

        # Discharge, Charge and Storage
        if t == 0:
            assert (np.abs(solution.Storage[t] - solution.StartCharge * solution.cphe * 1000) <= 1).all(), f"Storage Start incorrect. t: {t}"
        else:
            assert (np.abs(solution.Storage[t-1] - solution.Storage[t] 
                           + solution.Charge[t-1] * solution.resolution * solution.efficiency
                           - solution.Discharge[t-1] * solution.resolution) <= 1).all(), f"Storage dis/charge accounting incorrect. t: {t}"

    assert (np.amax(solution.Charge, axis=0)    - 1000*solution.cphp  <= 1).all(), "Storage charging exceeds bounds."
    assert (np.amax(solution.Discharge, axis=0) - 1000*solution.cphp  <= 1).all(), "Storage discharging exceeds bounds."
    assert (np.amax(solution.Storage, axis=0)   - 1000*solution.cphe  <= 1).all(), "Storage level exceeds bounds."
    assert (np.amin(solution.Storage, axis=0)                         >= -0.1).all(), "Storage level goes negative"
    assert (np.amax(solution.Hvdc, axis=0)      - 1000*solution.chvdc <= 1).all(), "Transmission exceeds line capacity."
    assert (np.amin(solution.Hvdc, axis=0)      + 1000*solution.chvdc >= -1).all(), "Transmission exceeds line capacity."
    assert (solution.Transmission.sum(axis=1) >= 0).all(), "DClosses are negative"

    try: assert ((solution.Charge > 0.1) * (solution.Discharge > 0.1)).sum() == 0 
    except AssertionError: warn("Simultaneous charging and discharging", BehaviourWarning)
    try: assert ((solution.Spillage > 0.1) * (solution.Discharge > 0.1)).sum() == 0
    except AssertionError: warn("Simultaneous discharge and spillage", BehaviourWarning)
    try: assert ((solution.Transmission < 0.1) * (solution.Spillage > 0.1)).sum() == 0
    except AssertionError: warn("Simultaneous import and spillage", BehaviourWarning)

    print('Debugging: everything is ok')

    return True


def LPGM(solution):
    """Load profiles and generation mix data"""

    C = np.vstack((solution.Load.sum(axis=1), solution.PV.sum(axis=1), solution.Wind.sum(axis=1),
                   solution.Hydro.sum(axis=1), solution.Bio.sum(axis=1), solution.Discharge.sum(axis=1),
                   -solution.Charge.sum(axis=1), -solution.Spillage.sum(axis=1),
                  solution.Storage.sum(axis=1), solution.Hvdc.T)).T
    C = np.around(C)

    datentime = np.array([(dt(solution.firstyear, 1, 1, 0, 0) + x * td(minutes=60 *
                         solution.resolution)).strftime('%a %d-%b %Y %H:%M') for x in range(solution.intervals)])
    C = np.insert(C.astype('str'), 0, datentime, axis=1)

    header = ','.join(['Date & time', 'Demand', 'Solar photovoltaics', 'Wind', 'Hydropower', 'Biomass',
                      'PHES-Discharge', 'PHES-Charge', 'Energy spillage', 'PHES-Storage'] +
                      [f'{solution.Nodel[n[0]]}-{solution.Nodel[n[1]]}' for n in solution.network]
                      )

    np.savetxt(f'Results/S{solution.scenario}.csv', C,fmt='%s', delimiter=',', header=header, comments='')

    if solution.scenario >= 21:
        header = ','.join(['Date & time', 'Demand', 'Solar photovoltaics', 'Wind', 'Hydropower', 'Biomass',
                          'PHES-Discharge', 'PHES-Charge', 'Transmission', 'Energy spillage', 'PHES-Storage'])

        for j in range(solution.nodes):
            C = np.vstack([solution.Load[:, j], solution.PV[:, j], solution.Wind[:, j], solution.Hydro[:, j], 
                           solution.Bio[:, j], solution.Discharge[:, j], -solution.Charge[:, j], 
                           solution.Transmission[:, j], -solution.Spillage[:, j], solution.Storage[:, j]]).T
            C = np.around(C)

            C = np.insert(C.astype('str'), 0, datentime, axis=1)
            np.savetxt(f'Results/S{solution.scenario}-{solution.Nodel[j]}.csv', C, fmt='%s', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True


def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""

    print('Levelised costs of electricity:')
    print('\u2022 LCOE:', solution.LCOE)
    print('\u2022 LCOG:', solution.LCOG)
    print('\u2022 LCOB:', solution.LCOB)
    print('\u2022 LCOG-PV:', solution.LCOGP, f'({solution.CFPV})' )
    print('\u2022 LCOG-Wind:', solution.LCOGW, f'({solution.CFWind})')
    print('\u2022 LCOG-Hydro:', solution.LCOGH)
    print('\u2022 LCOG-Bio:', solution.LCOGB)
    print('\u2022 LCOB-Storage:', solution.LCOBS)
    print('\u2022 LCOB-Transmission:', solution.LCOBT)
    print('\u2022 LCOB-Spillage & loss:', solution.LCOBL)

    D = np.atleast_2d(np.array(
        [solution.Energy, solution.cpv.sum(), solution.GPV, solution.cwind.sum(), solution.GWind, 
         solution.chydro.sum() + solution.cbio.sum(), solution.GHydro + solution.GBio, solution.cphp.sum(), solution.cphe.sum()]
              + list(solution.chvdc)
              + [solution.LCOE, solution.LCOG,solution.LCOBS, solution.LCOBT, solution.LCOBL]))

    header = ','.join(['Energy (PWh p.a.)', 'PV (GW)', 'PV (GWh p.a.)', 'Wind (GW)', 'Wind (GWh p.a.)',
                     'Hydro & Bio (GW)', 'Hydro & Bio (GWh p.a.)', 'Pumped Hydro capacity (GW)',
                     'Pumped Hydro capacity (GWh)'] +
                    [f'{solution.Nodel[n[0]]}-{solution.Nodel[n[1]]} (GW)' for n in solution.network] +
                    ['LCOE', 'LCOG', 'LCOB (storage)', 'LCOB (transmission)', 'LCOB (curtailment)']
                    )

    np.savetxt(f'Results/GGTA{solution.scenario}.csv', D, fmt='%f', delimiter=',', header=header, comments='')
    print('Energy generation, storage and transmission information is produced.')

    return True

def Information(solution):

    start=dt.now()
    print("Statistics start at", start)

    LPGM(solution)
    GGTA(solution)
    Debug(solution)


    end=dt.now()
    print("Statistics took", end - start)

    return True

if __name__ == '__main__':
    from Input import scenario, Solution
    from Optimisation import reconstruct_from_capacities
    capacities = np.genfromtxt(f"Results/Optimisation_resultx{scenario}.csv", delimiter=',')[1:]
    
    model = reconstruct_from_capacities(capacities)
    S = Solution(model)
    Information(S)
