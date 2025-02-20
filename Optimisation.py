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

#%%
def instantiate_model():
    print("Instantiating optimiser:", dt.now())
    model = pyo.ConcreteModel()
    
    adj_energy = (MLoad[:intervals, :].sum() * 1000 * resolution / years)
    
    model.pvl   = pyo.RangeSet(npv)
    model.onswl = pyo.RangeSet(nonsw)
    model.offwl = pyo.RangeSet(noffw)
    model.lines = pyo.RangeSet(nhvdc)
    model.nodes = pyo.RangeSet(nodes)
    model.t     = pyo.RangeSet(intervals) 
    
    model.hvdcCost = pyo.Param(
        model.lines, 
        domain=pyo.Reals, 
        initialize = lambda m, n: costs.hvdc[n-1]
        )
    
    model.cpv = pyo.Var(
        model.pvl,   
        domain=pyo.NonNegativeReals, 
        bounds=lambda _: (0, 20),
        initialize=lambda _i: 10,
        )
    model.consw = pyo.Var(
        model.onswl, 
        domain=pyo.NonNegativeReals, 
        bounds=lambda _: (0, 20),
        initialize=lambda _: 10,
        )
    model.coffw = pyo.Var(
        model.offwl, 
        domain=pyo.NonNegativeReals, 
        bounds=lambda _: (0, 20),
        initialize=lambda _: 10,
        )
    model.cgas = pyo.Var(
        model.nodes, 
        domain=pyo.NonNegativeReals,
        bounds=lambda _: (0, 0),
        initialize=lambda _: 0,
        )
    model.cphp = pyo.Var(
        model.nodes, 
        domain=pyo.NonNegativeReals, 
        bounds=lambda m, n: (0,20),#(contingency[n-1], 20),
        initialize=lambda m, n: 10,#(contingency[n-1]+10)/2,
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
        bounds=lambda _: (0,50),
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
        
    def expr_energy_balance(m, t, n):
        return (MLoad[t-1, n-1] 
                + m.charge[t,n] 
                - sum((m.cpv[z]  *TSPV[  t-1, z-1] for z in pv_zs_in_n[  n-1])) 
                - sum((m.consw[z]*TSOnsW[t-1, z-1] for z in onsw_zs_in_n[n-1]))
                - sum((m.coffw[z]*TSOffW[t-1, z-1] for z in offw_zs_in_n[n-1]))
                - m.hydro[t,n] 
                - m.bio[t,n] 
                - m.gas[t,n]
                - m.discharge[t,n] 
                + sum((m.hvdc_pos[t, l] - m.hvdc_neg[t, l]*(1-masked_DCloss[l-1]) for l in pos_export_lines[n-1]))
                + sum((m.hvdc_neg[t, l] - m.hvdc_pos[t, l]*(1-masked_DCloss[l-1]) for l in neg_export_lines[n-1]))
                )
    
    model.energy_balance = pyo.Expression(model.t, model.nodes, rule=expr_energy_balance)
    model.constr_energy_balance = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.energy_balance[t,n]<=0)
    
    model.CostPV    = pyo.Expression(rule=lambda m: pyo.summation(m.cpv)       * costs.pv)
    model.CostOnsW  = pyo.Expression(rule=lambda m: pyo.summation(m.consw)     * costs.onsw)
    model.CostOffW  = pyo.Expression(rule=lambda m: pyo.summation(m.coffw)     * costs.offw)
    model.CostGas   = pyo.Expression(rule=lambda m: pyo.summation(m.cgas)      * costs.gas[0] + 
                                                    pyo.summation(m.gas) * resolution / years * 1000 * costs.gas[1]) #GW -> MWh p.a.
    model.CostPH    = pyo.Expression(rule=lambda m: pyo.summation(m.cphp)      * costs.phes[0] +
                                                    pyo.summation(m.cphe)      * costs.phes[1] + 
                                                    pyo.summation(m.discharge) * resolution / years * 1000 * costs.phes[2] + 
                                                    costs.phes[3])
    model.CostHydro = pyo.Expression(rule=lambda m: pyo.summation(m.hydro)     * resolution / years * 1000 * costs.hydro)
    model.CostBio   = pyo.Expression(rule=lambda m: pyo.summation(m.bio)       * resolution / years * 1000 * (costs.hydro+0.1))
    model.CostDC    = pyo.Expression(rule=lambda m: pyo.summation(m.hvdcCost, m.chvdc))
    model.CostAC    = pyo.Expression(rule=lambda m: (
        pyo.summation(m.cpv) + 
        pyo.summation(m.consw) +
        pyo.summation(m.coffw) +
        pyo.summation(m.cgas) +
        0) * costs.ac
        )
    
    model.LCOE = pyo.Expression(rule=lambda m: (
        m.CostPV + 
        m.CostOnsW + 
        m.CostOffW + 
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
    model.consw.fix()
    model.coffw.fix()
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
    """ minimise instances of simulataneous dis/charging, im/exporting, etc."""
    model = fix_investment(model)
    
    model.operations = pyo.Expression(rule=lambda m: (
        pyo.summation(m.charge)
        + pyo.summation(m.discharge) 
        + pyo.summation(m.hvdc_pos)
        + pyo.summation(m.hvdc_neg)
        - pyo.summation(m.energy_balance)
        ))
    
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
    for i, v in enumerate(model.cpv.values()):
        v.fix(capacities[        i-1])
    for i, v in enumerate(model.consw.values()):
        v.fix(capacities[pidx   +i-1])
    for i, v in enumerate(model.coffw.values()):
        v.fix(capacities[onswidx+i-1])
    for i, v in enumerate(model.cphp.values()):
        v.fix(capacities[offwidx+i-1])
    for i, v in enumerate(model.cphe.values()):
        v.fix(capacities[spidx  +i-1])
    for i, v in enumerate(model.chvdc.values()):
        v.fix(capacities[seidx  +i-1])
    for i, v in enumerate(model.cgas.values()):
        v.fix(capacities[hvidx  +i-1])

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
    
    print('pv:',        S.cpv_n)
    print('ons wind:',  S.consw_n)
    print('offs wind:', S.coffw_n)
    print('gas:',       S.cgas)
    print('php:',       S.cphp)
    print('phe:',       S.cphe)
    print('chvdc:',     S.chvdc)
    
    try:
        np.savetxt(f'Results/Optimisation_resultx{scenario}.csv', S.x.reshape(1,-1), fmt='%s', delimiter=',')
    except FileNotFoundError:
        import os 
        os.mkdir('Results')
        np.savetxt(f'Results/Optimisation_resultx{scenario}.csv', S.x.reshape(1,-1), fmt='%s', delimiter=',')

    from Statistics import Information
    Information(S)
