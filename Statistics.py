# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from warnings import warn
from Input import TSPV, TSOnsW, TSOffW, costs

class BehaviourWarning(Warning):
    def __init__(self, message):
        self.message=message
    def __str__(self):
        return repr(self.message)

def zero_safe_divide(numerator, denominator, retval=0):
    return numerator / denominator if denominator != 0 else retval

def zero_safe_divide_arr(numerator, denominator, retval=0.0):
    retarr = np.divide(numerator, denominator, where=denominator != 0)
    retarr[denominator == 0] = retval
    return retarr

def Debug(solution):
    """Debugging"""

    for t in range(solution.intervals):
        # supply-demand
        sup_dem = (
            solution.Load[t] 
            + solution.Spillage[t] 
            + solution.Charge[t]
            - solution.Discharge[t] 
            - solution.Hydro[t] 
            - solution.Bio[t]
            - solution.Transmission[t] 
            - solution.PV[t] 
            - solution.OnsW[t]
            - solution.OffW[t]
            - solution.Gas[t]
            )
        
        assert (np.abs(sup_dem) <= 1).all(), f"Supply demand unbalanced. t: {t}. supply-demand: {-sup_dem}"

        # Discharge, Charge and Storage
        if t == 0:
            assert (np.abs(solution.Storage[t] - solution.StartCharge * solution.cphe * 1000) <= 1).all(), f"Storage Start incorrect. t: {t}"
        else:
            assert (np.abs(solution.Storage[t-1] - solution.Storage[t] 
                           + solution.Charge[t-1] * solution.resolution * solution.efficiency
                           - solution.Discharge[t-1] * solution.resolution) <= 1).all(), f"Storage dis/charge accounting incorrect. t: {t}"

    assert (np.amax(solution.Charge, axis=0)    - 1000*solution.cphp  <= 1).all(),    "Storage charging exceeds bounds."
    assert (np.amax(solution.Discharge, axis=0) - 1000*solution.cphp  <= 1).all(),    "Storage discharging exceeds bounds."
    assert (np.amax(solution.Storage, axis=0)   - 1000*solution.cphe  <= 1).all(),    "Storage level exceeds bounds."
    assert (np.amin(solution.Storage, axis=0)                         >= -0.1).all(), "Storage level goes negative"
    assert (np.amax(solution.Hvdc, axis=0)      - 1000*solution.chvdc <= 1).all(),    "Transmission exceeds line capacity."
    assert (np.amin(solution.Hvdc, axis=0)      + 1000*solution.chvdc >= -1).all(),   "Transmission exceeds line capacity."
    assert (solution.Transmission.sum(axis=1) <= 0).all(), "DClosses are negative"

    try: assert ((solution.Charge > 0.1) * (solution.Discharge > 0.1)).sum() == 0 
    except AssertionError: warn("Simultaneous charging and discharging. (may artificially reduce LCOBS)", BehaviourWarning)
    try: assert ((solution.Spillage > 0.1) * (solution.Discharge > 0.1)).sum() == 0
    except AssertionError: warn("Simultaneous discharge and spillage (may artificially inflate LCOBS and reduce LCOBL)", BehaviourWarning)
    try: assert ((solution.Transmission > 0.1) * (solution.Spillage > 0.1)).sum() == 0
    except AssertionError: warn("Simultaneous import and spillage (may artificially inflate LCOBL and reduce LCOBT)", BehaviourWarning)

    print('Debugging: everything is ok')

    return True


def LPGM(solution):
    """Load profiles and generation mix data"""

    C = np.vstack((
        solution.Load.sum(axis=1), 
        solution.PV.sum(axis=1), 
        solution.OnsW.sum(axis=1),
        solution.OffW.sum(axis=1),
        solution.Hydro.sum(axis=1), 
        solution.Bio.sum(axis=1), 
        solution.Discharge.sum(axis=1),
        -solution.Charge.sum(axis=1),
        -solution.Spillage.sum(axis=1),
        solution.Storage.sum(axis=1), 
        solution.Hvdc.T)
        ).T
    C = np.around(C)

    datentime = np.array([(dt(solution.firstyear, 1, 1, 0, 0) + x * td(minutes=60 *
                         solution.resolution)).strftime('%a %d-%b %Y %H:%M') for x in range(solution.intervals)])
    C = np.insert(C.astype('str'), 0, datentime, axis=1)

    header = ','.join(['Date & time', 'Demand', 'Solar photovoltaics', 'Onshore Wind', 'Offshore Wind', 'Hydropower',
                       'Biomass', 'PHES-Discharge', 'PHES-Charge', 'Energy spillage', 'PHES-Storage'] +
                      [f'{solution.Nodel[n[0]]}-{solution.Nodel[n[1]]}' for n in solution.network])

    np.savetxt(f'Results/S{solution.scenario}.csv', C,fmt='%s', delimiter=',', header=header, comments='')

    if solution.scenario >= 21:
        header = ','.join(['Date & time', 'Demand', 'Solar photovoltaics', 'Onshore Wind', 'Offshore Wind', 'Hydropower', 
                           'Biomass', 'PHES-Discharge', 'PHES-Charge', 'Transmission', 'Energy spillage', 'PHES-Storage'])

        for j in range(solution.nodes):
            C = np.vstack((
                solution.Load[:, j], 
                solution.PV[:, j], 
                solution.OnsW[:, j], 
                solution.OffW[:, j], 
                solution.Hydro[:, j], 
                solution.Bio[:, j], 
                solution.Discharge[:, j], 
                -solution.Charge[:, j], 
                -solution.Transmission[:, j], 
                -solution.Spillage[:, j], 
                solution.Storage[:, j])
                ).T
            C = np.around(C)

            C = np.insert(C.astype('str'), 0, datentime, axis=1)
            np.savetxt(f'Results/S{solution.scenario}-{solution.Nodel[j]}.csv', C, fmt='%s', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True


def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""

    print('Levelised costs of electricity:')
    print(f'\u2022 LCOE: {solution.LCOE}')
    print(f'\u2022 LCOG: {solution.LCOG}')
    print(f'\u2022 LCOB: {solution.LCOB}')
    print(f'\u2022 LCOG-PV: {solution.LCOGP} (CF: {solution.CFPV}%)')
    print(f'\u2022 LCOG-Onshore Wind: {solution.LCOGOnsW} (CF: {solution.CFOnsW}%)')
    print(f'\u2022 LCOG-Offshore Wind: {solution.LCOGOffW} (CF: {solution.CFOffW}%)')
    print(f'\u2022 LCOG-Gas: {solution.LCOGG} (CF: {solution.CFGas}%)')
    print(f'\u2022 LCOG-Hydro: {solution.LCOGH}')
    print(f'\u2022 LCOG-Bio: {solution.LCOGB}')
    print(f'\u2022 LCOB-Storage: {solution.LCOBS}')
    print(f'\u2022 LCOB-Transmission: {solution.LCOBT}')
    print(f'\u2022 LCOB-Spillage & loss: {solution.LCOBL}')

    D = np.atleast_2d(np.array(
        [solution.Energy, solution.cpv.sum(), solution.GPV, solution.consw.sum(), solution.GOnsW, 
         solution.coffw.sum(), solution.GOffW, solution.chydro.sum() + solution.cbio.sum(), 
         solution.GHydro.sum() + solution.GBio.sum(), solution.cphp.sum(), solution.cphe.sum(), solution.GPHES.sum()]
        + list(solution.chvdc)
        + [solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL]))

    header = ','.join(['Energy (PWh p.a.)', 'Utility PV (GW)', 'Utility PV (TWh p.a.)', 
                       'Onshore Wind (GW)', 'Onshore Wind (TWh p.a.)', 'Offshore Wind (GW)', 
                       'Offshore Wind (TWh p.a.)', 'Hydro & Bio (GW)', 'Hydro & Bio (TWh p.a.)', 
                       'PHES capacity (GW)', 'PHES capacity (GWh)', 'PHES (TWh p.a.)'] +
                      [f'{solution.Nodel[n[0]]}-{solution.Nodel[n[1]]} (GW)' for n in solution.network] +
                      ['LCOE', 'LCOG', 'LCOB (storage)', 'LCOB (transmission)', 'LCOB (curtailment)']
                      )

    np.savetxt(f'Results/GGTA{solution.scenario}.csv', D, fmt='%f', delimiter=',', header=header, comments='')
    print('Energy generation, storage and transmission information is produced.')

    return True

def GBTO(solution):
    #%%
    """Grid-balancing trade offs"""
    """There are 4 strategies for grid balancing
        * Storage dis/charging
        * interstate transmission
        * load-shedding/curtailment
        * intra-state spatial diversity
        
    I have chosen not to view flexible generation as grid balancing
        # * flexible generation
    """
    
    EGBS = solution.Discharge.sum(axis=0), solution.Charge.sum(axis=0)
    EGBT = np.maximum(solution.Transmission, 0).sum(axis=0), -np.minimum(solution.Transmission, 0).sum(axis=0)
    EGBC = np.zeros(solution.nodes), solution.Spillage.sum(axis=0)
    EGBF = solution.Hydro.sum(axis=0) + solution.Gas.sum(axis=0) + solution.Bio.sum(axis=0), np.zeros(solution.nodes)
    
    EGBS, EGBT, EGBC, EGBF= (tuple(quantity * solution.resolution /solution.years for quantity in GB)
         for GB in (EGBS, EGBT, EGBC, EGBF)) # MWh per annum
    
    state_network = np.array([n in solution.network[i] for n in range(solution.nodes) for i in range(len(solution.network))]
                             ).reshape(solution.nodes, len(solution.network))
    CGBT = np.zeros(solution.nodes)
    for n in range(solution.nodes):
        # half the cost of each HVDC connection
        CGBT[n] += solution.CostDC[state_network[n]].sum()/2 
        # half the cost of energy lost to efficiency
        CGBT[n] += (np.abs(solution.Hvdc[:, state_network[n]] * solution.DCloss[state_network[n]]).sum() 
                    * solution.resolution / solution.years * solution.LCOG) / 2
        CGBT[n] /= (EGBT[0][n]+EGBT[1][n])
        
    # (capital cost of storage + cost of efficiency loss) / abs MWh balanced 
    CGBS = (solution.CostPH + solution.LCOG * (EGBS[1] - EGBS[0])) / (EGBS[0] + EGBS[1]) 
    
    # cost of the energy shed
    CGBC = ((solution.CostPV + solution.CostOnsW + solution.CostOffW + 
             solution.CostHydro + solution.CostBio + solution.CostGas) /
             (solution.GPV + solution.GOnsW + solution.GOffW + solution.GHydro + solution.GBio + solution.GGas))
    
    # Weighted average of costs of generation of flexible resources used 
    CGBF = ((solution.CostHydro * solution.Hydro.sum(axis=0) + 
             solution.CostBio   * solution.Bio.sum(axis=0) + 
             solution.CostGas   * solution.Gas.sum(axis=0)) / 
            (solution.Hydro + solution.Bio + solution.Gas).sum(axis=0) /
            (solution.GGas + solution.GHydro + solution.GBio))

    cfpv, cfonsw, cfoffw = (TS[:solution.intervals, :].mean(axis=0) for TS in (TSPV, TSOnsW, TSOffW))
    EGBDpv = np.zeros(solution.nodes), np.zeros(solution.nodes)
    EGBDonsw = np.zeros(solution.nodes), np.zeros(solution.nodes)
    EGBDoffw = np.zeros(solution.nodes), np.zeros(solution.nodes)
    CGBDpv, CGBDonsw, CGBDoffw = np.zeros(solution.nodes), np.zeros(solution.nodes), np.zeros(solution.nodes)

    for i, node in enumerate(solution.Nodel):
        nodemask = solution.PVl == node
        if nodemask.sum() > 0:
            cfpv_z = cfpv[nodemask]
            best_z, best = cfpv_z.argmax(), cfpv_z.max()
            diversity = TSPV[:solution.intervals, nodemask]
            diversity = diversity - np.atleast_2d(diversity[:, best_z]).T
            surplus =  np.maximum(diversity, 0).sum(axis=0) * solution.cpv[nodemask] * 1000. * solution.resolution / solution.years
            deficit = -np.minimum(diversity, 0).sum(axis=0) * solution.cpv[nodemask] * 1000. * solution.resolution / solution.years
            EGBDpv[0][i] = surplus.sum()
            EGBDpv[1][i] = deficit.sum()
            #cost of extra power capacity to produce same amount of energy compared to using just best site
            CGBDpv[i] = zero_safe_divide(
                (solution.cpv[nodemask].sum() - (solution.cpv[nodemask] * cfpv_z).sum() / best) * costs.pv,
                surplus.sum() + deficit.sum())
        
        nodemask = solution.OnsWl == node
        if nodemask.sum() > 0:
            cfonsw_z = cfonsw[nodemask]
            best_z, best = cfonsw_z.argmax(), cfonsw_z.max()
            diversity = TSOnsW[:solution.intervals, nodemask]
            diversity = diversity - np.atleast_2d(diversity[:, best_z]).T
            surplus =  np.maximum(diversity, 0).sum(axis=0) * solution.consw[nodemask] * 1000. * solution.resolution / solution.years
            deficit = -np.minimum(diversity, 0).sum(axis=0) * solution.consw[nodemask] * 1000. * solution.resolution / solution.years
            EGBDonsw[0][i] = surplus.sum()
            EGBDonsw[1][i] = deficit.sum()
            #cost of extra power capacity to produce same amount of energy compared to using just best site
            CGBDonsw[i] = zero_safe_divide(
                (solution.consw[nodemask].sum() - (solution.consw[nodemask] * cfonsw_z).sum() / best) * costs.onsw,
                surplus.sum() + deficit.sum())

        
        nodemask = solution.OffWl == node
        if nodemask.sum() > 0:
            cfoffw_z = cfoffw[nodemask]
            best_z, best = cfoffw_z.argmax(), cfoffw_z.max()
            diversity = TSOffW[:solution.intervals, nodemask]
            diversity = diversity - np.atleast_2d(diversity[:, best_z]).T
            surplus =  np.maximum(diversity, 0).sum(axis=0) * solution.coffw[nodemask] * 1000. * solution.resolution / solution.years
            deficit = -np.minimum(diversity, 0).sum(axis=0) * solution.coffw[nodemask] * 1000. * solution.resolution / solution.years
            EGBDoffw[0][i] = surplus.sum()
            EGBDoffw[1][i] = deficit.sum()
            
            CGBDoffw[i] = zero_safe_divide(
                (solution.coffw[nodemask].sum() - (solution.coffw[nodemask] * cfoffw_z).sum() / best) * costs.offw, 
                surplus.sum() + deficit.sum())

    EGBD = (sum((egbd[0] for egbd in (EGBDpv, EGBDonsw, EGBDoffw))),
            sum((egbd[1] for egbd in (EGBDpv, EGBDonsw, EGBDoffw))))
    CGBD = zero_safe_divide_arr(
        sum((cgbd * sum(egbd) for cgbd, egbd in zip((CGBDpv, CGBDonsw, CGBDoffw), (EGBDpv, EGBDonsw, EGBDoffw)))),
        sum((sum(egbd) for egbd in (EGBDpv, EGBDonsw, EGBDoffw))))

    def add_system_cgb(egb, cgb):
        return np.append(cgb, (cgb * zero_safe_divide(sum(egb), sum(egb).sum(), np.zeros(solution.nodes))).sum())
    def add_system_egb(egb):
        return (np.append(egb[0], egb[0].sum()), np.append(egb[1], egb[1].sum()))

    CGBS = add_system_cgb(EGBS, CGBS)
    CGBT = add_system_cgb(EGBT, CGBT)
    CGBC = add_system_cgb(EGBC, CGBC)
    # CGBF = add_system_cgb(EGBF, CGBF)
    CGBDpv = add_system_cgb(EGBDpv, CGBDpv)
    CGBDonsw = add_system_cgb(EGBDonsw, CGBDonsw)
    CGBDoffw = add_system_cgb(EGBDoffw, CGBDoffw)
    CGBD = add_system_cgb(EGBD, CGBD)
    
    EGBS = add_system_egb(EGBS)
    EGBT = add_system_egb(EGBT)
    EGBC = add_system_egb(EGBC)
    # EGBF = add_system_egb(EGBF)
    EGBDpv = add_system_egb(EGBDpv)
    EGBDonsw = add_system_egb(EGBDonsw)
    EGBDoffw = add_system_egb(EGBDoffw)
    EGBD = add_system_egb(EGBD)
    
    LCOB = (CGBS * sum(EGBS) + CGBT * sum(EGBT) + CGBC * sum(EGBC) + CGBD * sum(EGBD)) / sum((sum(EGBS), sum(EGBT), sum(EGBC), sum(EGBD)))
    
    EGB = np.stack((
        LCOB,
        *EGBS, CGBS, 
        *EGBT, CGBT, 
        *EGBC, CGBC, 
        # *EGBF, CGBF, 
        *EGBDpv, CGBDpv, 
        *EGBDonsw, CGBDonsw, 
        *EGBDoffw, CGBDoffw,
        *EGBD, CGBD, 
        )).T
    EGB = np.insert(EGB.astype('str'), 0, np.append(solution.Nodel, 'system'), axis=1)

    header = ','.join(['State', 'Levelised Cost of Balancing',
        'Storage Deficit Filling', 'Storage Surplus Shaving', 'Storage Cost of Balancing',
        'Transmission Deficit Filling', 'Transmission Surplus Shaving', 'Transmission Cost of Balancing',
        'Curtailment Deficit Filling', 'Curtailment Surplus Shaving', 'Curtailment Cost of Balancing',
        # 'Flexible Deficit Filling', 'Flexible Surplus Shaving', 'Flexible Cost of Balancing',
        'Diversity (pv) Deficit Filling', 'Diversity (pv) Surplus Shaving', 'Diversity (pv) Cost of Balancing',
        'Diversity (onsw) Deficit Filling', 'Diversity (onsw) Surplus Shaving', 'Diversity (onsw) Cost of Balancing',
        'Diversity (offw) Deficit Filling', 'Diversity (offw) Surplus Shaving', 'Diversity (offw) Cost of Balancing',
        'Diversity Deficit Filling', 'Diversity Surplus Shaving', 'Diversity Cost of Balancing',
        ])

    np.savetxt(f'Results/EGB{solution.scenario}.csv', EGB, fmt='%s', delimiter=',', header=header, comments='')

#%%


def Information(solution):

    start=dt.now()
    print("Statistics start at", start)
    Debug(solution)
    LPGM(solution)
    GGTA(solution)
    GBTO(solution)
    end=dt.now()
    print("Statistics took", end - start)

    return True

#%%
if __name__ == '__main__':
    from Input import scenario, Solution
    from Optimisation import reconstruct_from_capacities
    capacities = np.genfromtxt(f"Results/Optimisation_resultx{scenario}.csv", delimiter=',')[1:]
    
    model = reconstruct_from_capacities(capacities)
    S = Solution(model)
    Information(S)
