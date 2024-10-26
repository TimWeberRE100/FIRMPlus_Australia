# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from datetime import datetime as dt
from time import perf_counter

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from Input import * 

MLoad = MLoad / 1000. # MW to GW

# =============================================================================
# Be careful about Nodel or np.unique(PVl/Windl) when zones don't have a technology
# =============================================================================
pv_zs_in_n = [np.where(PVl==node)[0] + 1 for node in Nodel] 
wind_zs_in_n = [np.where(Windl==node)[0] + 1 for node in Nodel]

import_lines = [np.where(network[:,0]==n)[0] + 1 for n in range(nodes)]
export_lines = [np.where(network[:,1]==n)[0] + 1 for n in range(nodes)]

if scenario >= 21:
    CostPH, CostDC = -1, -1
else:
    CostPH, CostDC = 0,0 

nyears = 1

leapdays = (nyears+(4-59/365))//4

ndays = 365*nyears + leapdays



#%%
print("Instantiating model:", dt.now())
model = pyo.ConcreteModel()

model.pvl = pyo.RangeSet(len(PVl))
model.windl = pyo.RangeSet(len(Windl))
model.lines = pyo.RangeSet(len(network))
model.nodes = pyo.RangeSet(nodes)

model.t = pyo.RangeSet(48*ndays) 

model.cpv = pyo.Var(
    model.pvl,   
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, len(PVl)+1), zip(len(PVl)*[0.], len(PVl)*[50.]))),
    initialize=dict(zip(range(1, len(PVl)+1), len(PVl)*[10.])),
    )
model.cwind = pyo.Var(
    model.windl, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, len(Windl)+1), zip(len(Windl)*[0.], len(Windl)*[50.]))),
    initialize=dict(zip(range(1, len(Windl)+1)  , len(Windl)*[10.])),
    )
model.cphp = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(nodes*[0.], nodes*[50.]))),
    initialize=dict(zip(range(1, nodes+1), nodes*[10.])),
    )
model.cphs = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(nodes*[0.], nodes*[500.]))),
    initialize=dict(zip(range(1, nodes+1), nodes*[100.])),
    )
model.chvdc = pyo.Var(
    model.lines, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, len(network)+1), zip(len(network)*[0.], len(network)*[100.]))),
    initialize=dict(zip(range(1, len(network)+1), len(network)*[50.])),
    )

model.hvdcCost = pyo.Param(model.lines, domain=pyo.Reals, initialize = dict(zip(range(1, len(network)+1), factor[4:11][network_mask])))

model.charge =  pyo.Var(model.t, model.nodes, domain=pyo.Reals)
model.storage = pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.hvdc =    pyo.Var(model.t, model.lines, domain=pyo.Reals)
model.hydro =   pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.bio =     pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.spillage =pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)

model.constr_charge_power_lower = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n:-m.cphp[n] <= m.charge[t, n])
model.constr_charge_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.charge[t, n] <= m.cphp[n])

model.constr_storage_energy_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.storage[t, n] <= m.cphs[n])

model.constr_hydro_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.hydro[t, n] <= CHydro[n-1])
model.constr_bio_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.bio[t, n] <= CBio[n-1])

model.constr_hvdc_line_power_lower = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l: sum((m.hvdc[t, l] for n in m.nodes)) >= -m.chvdc[l])
model.constr_hvdc_line_power_upper = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l: sum((m.hvdc[t, l] for n in m.nodes)) <= m.chvdc[l])

model.constr_max_hydro = pyo.Constraint(rule=lambda m: pyo.summation(model.hydro) <= 20_000 * nyears)


def constr_state_of_charge(m, t, n):
    if t==1:
        return m.storage[t, n] == 0.5 * m.cphs[n]
    else:
        return m.storage[t, n] == m.storage[t-1, n] - m.charge[t-1, n] * resolution * efficiency

model.constr_storage_state_of_charge = pyo.Constraint(model.t, model.nodes, rule=constr_state_of_charge)


def constr_power_balance_lower(m, t, n):
    return (MLoad[t-1, n-1] + m.spillage[t, n] 
            - sum((m.cpv[z]*TSPV[t-1, z-1] for z in pv_zs_in_n[n-1])) - sum((m.cwind[z]*TSWind[t-1, z-1] for z in wind_zs_in_n[n-1]))
             - m.hydro[t,n] - m.bio[t,n] - m.charge[t,n] 
            - sum((m.hvdc[t, l] for l in import_lines[n-1])) + sum((m.hvdc[t, l] for l in export_lines[n-1]))
            ) >= -0.001
def constr_power_balance_upper(m, t, n):
    return (MLoad[t-1, n-1] + m.spillage[t, n] 
            - sum((m.cpv[z]*TSPV[t-1, z-1] for z in pv_zs_in_n[n-1])) - sum((m.cwind[z]*TSWind[t-1, z-1] for z in wind_zs_in_n[n-1]))
            - m.hydro[t,n] - m.bio[t,n] - m.charge[t,n] 
            - sum((m.hvdc[t, l] for l in import_lines[n-1])) + sum((m.hvdc[t, l] for l in export_lines[n-1]))
            ) <= 0.001

model.constr_power_balance_upper = pyo.Constraint(model.t, model.nodes, rule=constr_power_balance_upper)
model.constr_power_balance_lower = pyo.Constraint(model.t, model.nodes, rule=constr_power_balance_lower)

def objective(m):
    cost = (
        factor[0] * pyo.summation(m.cpv) +
        factor[1] * pyo.summation(m.cwind) + 
        factor[2] * pyo.summation(m.cphp) +
        factor[3] * pyo.summation(m.cphs) + 
        pyo.summation(m.hvdcCost, m.chvdc) +
        factor[11] * pyo.summation(m.cpv) +
        factor[12] * pyo.summation(m.cwind) + 
        factor[13] * pyo.summation(m.hydro)/ 1000. +
        (factor[13]+0.000_001) * pyo.summation(m.bio)/ 1000. + # use hydro first
        factor[14] * CostPH +
        factor[15] * CostDC +

        0)
    LCOE = cost/energy
    #HVDC loss not currently included. Suggest including it in energy balance
    
    # Penalties 
    
    return LCOE #+ Penalties
    

model.OBJ = pyo.Objective(rule=objective)


opt = pyo.SolverFactory('gurobi')

start=dt.now()
print("Optimisation starts:", start)
opt.solve(model)
end=dt.now()
print("Optimisation took:", end-start)



model.OBJ.display()

Charge = np.array([model.charge[i].value for i in model.charge]).reshape(-1, nodes)
Storage = np.array([model.storage[i].value for i in model.storage]).reshape(-1, nodes)
Hydro = np.array([model.hydro[i].value for i in model.hydro]).reshape(-1, nodes)

        

