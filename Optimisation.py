# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from datetime import datetime as dt
import csv
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from Input import * 
from Costs import *



#%%
def instantiate_model():
    print("Instantiating optimiser:", dt.now())
    model = pyo.ConcreteModel()
    
    adj_energy = (MLoad[:intervals, :].sum() * pow(10,3) * resolution / years)
    
    model.pvl   = pyo.RangeSet(npv)
    model.windl = pyo.RangeSet(nwind)
    model.lines = pyo.RangeSet(nhvdc)
    model.nodes = pyo.RangeSet(nodes)
    model.t     = pyo.RangeSet(intervals) 
    model.tn = pyo.Set(model.nodes, within=model.t, initialize={n:list(model.t) for n in model.nodes})
    
    model.hvdcCost = pyo.Param(
        model.lines, 
        domain=pyo.Reals, 
        initialize = lambda m, n: transmission_costs[n-1]+substation_costs
        )
    
    model.cpv = pyo.Var(
        model.pvl,   
        domain=pyo.NonNegativeReals, 
        bounds=lambda _: (0, 20),
        initialize=lambda _i: 10,
        )
    model.cwind = pyo.Var(
        model.windl, 
        domain=pyo.NonNegativeReals, 
        bounds=lambda _: (0, 20),
        initialize=lambda _: 10,
        )
    model.cgas = pyo.Var(
        model.nodes, 
        domain=pyo.NonNegativeReals,
        bounds=lambda _: (0, 20),
        initialize=lambda _: 10,
        )
    model.cphp = pyo.Var(
        model.nodes, 
        domain=pyo.NonNegativeReals, 
        bounds=lambda m, n: (contingency[n-1], 20),
        initialize=lambda m, n: (contingency[n-1]+10)/2,
        )
    model.cphe = pyo.Var(
        model.nodes, 
        domain=pyo.NonNegativeReals, 
        bounds=lambda _: (0, 200),
        initialize=lambda _: 100,
        )
    model.chvdc = pyo.Var(
        model.lines, 
        domain=pyo.NonNegativeReals, 
        bounds=lambda _: (0,20),
        initialize=lambda _: 10,
        )
    
    model.charge =  pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
    model.discharge=pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
    model.storage = pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
    model.hydro =   pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
    model.bio =     pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
    model.gas =     pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
    
    model.hvdc_pos = pyo.Var(model.t, model.lines, domain=pyo.NonNegativeReals)
    model.hvdc_neg = pyo.Var(model.t, model.lines, domain=pyo.NonNegativeReals)
    
    model.constr_charge_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.charge[t, n] <= m.cphp[n])
    model.constr_discharge_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.discharge[t, n] <= m.cphp[n])
    model.constr_storage_energy_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.storage[t, n] <= m.cphe[n])
    
    model.constr_hydro_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.hydro[t, n] <= CHydro[n-1])
    model.constr_bio_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.bio[t, n] <= CBio[n-1])
    model.constr_gas_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.gas[t, n] <= m.cgas[n])
    
    model.constr_hvdc_power_import = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l: m.hvdc_pos[t, l] <= m.chvdc[l])
    model.constr_hvdc_power_export = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l: m.hvdc_neg[t, l] <= m.chvdc[l])
    
    model.constr_max_hydrobio = pyo.Constraint(rule=lambda m: pyo.summation(m.hydro)*0.001*resolution/years
                                               + pyo.summation(m.bio)*0.001*resolution/years <= 20.0) #TWh p.a.
    

    model.constr_gas_peaking_CF = pyo.Constraint(model.nodes, rule=lambda m, n: sum((m.gas[t,n] for t in m.t)) * resolution / years <= 0.2 * m.cgas[n] * 8760)

    def constr_state_of_charge(m, t, n):
        if t==1:
            return m.storage[t, n] == StartCharge * m.cphe[n]
        else:
            return m.storage[t, n] == m.storage[t-1, n] - m.discharge[t-1, n] * resolution + m.charge[t-1, n] * resolution * efficiency
    
    model.constr_storage_state_of_charge = pyo.Constraint(model.t, model.nodes, rule=constr_state_of_charge)
    
        
    def constr_power_balance(m, t, n):
        return (MLoad[t-1, n-1] 
                + m.charge[t,n] 
                - sum((m.cpv[z]*TSPV[t-1, z-1] for z in pv_zs_in_n[n-1])) 
                - sum((m.cwind[z]*TSWind[t-1, z-1] for z in wind_zs_in_n[n-1]))
                - m.hydro[t,n] 
                - m.bio[t,n] 
                - m.gas[t,n]
                - m.discharge[t,n] 
                + sum((m.hvdc_pos[t, l] - m.hvdc_neg[t, l]*(1-masked_DCloss[l-1]) for l in pos_export_lines[n-1]))
                + sum((m.hvdc_neg[t, l] - m.hvdc_pos[t, l]*(1-masked_DCloss[l-1]) for l in neg_export_lines[n-1]))
                ) <= 0.0
    
    model.constr_power_balance = pyo.Constraint(model.t, model.nodes, rule=constr_power_balance)
    
    model.CostPV    = pyo.Expression(rule=lambda m: pyo.summation(m.cpv)       * pv_costs)
    model.CostWind  = pyo.Expression(rule=lambda m: pyo.summation(m.cwind)     * wind_costs)
    model.CostGas   = pyo.Expression(rule=lambda m: pyo.summation(m.cgas) * gas_costs[0] + 
                                                    pyo.summation(m.gas) * resolution / years * 1000 * gas_costs[1]) #GW -> MWh p.a.
    model.CostPH    = pyo.Expression(rule=lambda m: pyo.summation(m.cphp)      * storage_costs[0] +
                                                    pyo.summation(m.cphe)      * storage_costs[1] + 
                                                    pyo.summation(m.discharge) * resolution / years * 1000 * storage_costs[2] + 
                                                    storage_costs[3])
    model.CostHydro = pyo.Expression(rule=lambda m: pyo.summation(m.hydro)     * resolution / years * 1000 * hydro_cost)
    model.CostBio   = pyo.Expression(rule=lambda m: pyo.summation(m.bio)       * resolution / years * 1000 * (hydro_cost+0.1))
    model.CostDC    = pyo.Expression(rule=lambda m: pyo.summation(m.hvdcCost, m.chvdc))
    model.CostAC    = pyo.Expression(rule=lambda m: (
        pyo.summation(m.cpv) + 
        pyo.summation(m.cwind) +
        pyo.summation(m.cgas) +
        0) * ACgen_costs
        )
    
    model.LCOE = pyo.Expression(rule=lambda m: (
        m.CostPV + 
        m.CostWind + 
        m.CostGas +
        m.CostPH + 
        m.CostHydro + 
        m.CostBio +
        m.CostDC + 
        m.CostAC
        ) / adj_energy
        )

    return model 

def fix_investment(model):
    model.cpv.fix()
    model.cwind.fix()
    model.cgas.fix()
    model.cphp.fix()
    model.cphe.fix()
    model.chvdc.fix()
    model.hydro.fix()
    model.bio.fix()
    model.gas.fix()
    return model

def unfix(model):
    model.unfix_all_vars()
    return model

def optimise_operations(model):
    model = fix_investment(model)
    
    model.operations = pyo.Expression(rule=lambda m: (pyo.summation(m.charge) +
        pyo.summation(m.discharge) + pyo.summation(m.hvdc_pos) + pyo.summation(m.hvdc_neg)))
    
    model.sensible_operations = pyo.Objective(rule=lambda m: m.operations)
    
    start=dt.now()
    print("Tuning operations. Start:",start)
    optimiser = pyo.SolverFactory('gurobi')
    optimiser.solve(model)
    end=dt.now()
    print("Tuning took", end-start)
    
    model.sensible_operations.deactivate()
    return model
    
def cost_optimise(model):
    model.least_cost = pyo.Objective(rule=lambda m: m.LCOE)
    
    start=dt.now()
    print("Optimisation starts:", start)
    optimiser = pyo.SolverFactory('gurobi')
    optimiser.solve(model)
    end=dt.now()
    print("Optimisation took:", end-start)
    
    model.least_cost.deactivate()
    return model

def reconstruct_from_capacities(capacities):
    model = instantiate_model()
    for v in model.cpv.values():
        v.fix(capacities[      i-1])
    for i in model.cwind:
        v.fix(capacities[pidx +i-1])
    for i in model.cphp:
        v.fix(capacities[widx +i-1])
    for i in model.cphe:
        v.fix(capacities[spidx+i-1])
    for i in model.chvdc:
        v.fix(capacities[seidx+i-1])

    model = cost_optimise(model)
    model = optimise_operations(model)
    return model

if __name__ == '__main__':
    model = instantiate_model()
    model = cost_optimise(model)
    model = optimise_operations(model)
    model.LCOE.display()
    
    #%%
    
    S = Solution(model, years)
    
    print('pv:', S.cpv_n)
    print('wind:', S.cwind_n)
    print('gas:', S.cgas)
    print('php:', S.cphp)
    print('phe:', S.cphe)
    print('chvdc:', S.chvdc)
    
    try:
        with open(f'Results/Optimisation_resultx{scenario}.csv', 'w', newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([S.LCOE] + list(S.cpv) + list(S.cwind) + list(S.cphp) + list(S.cphe) + list(S.chvdc))
    except FileNotFoundError:
        import os 
        os.mkdir('Results')
        with open(f'Results/Optimisation_resultx{scenario}.csv', 'w', newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([S.LCOE] + list(S.cpv) + list(S.cwind) + list(S.cphp) + list(S.cphe) + list(S.chvdc))
    
    from Statistics import Information
    Information(S)
