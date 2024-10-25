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

CHydro = CHydro-CBaseload

MLoad = MLoad / 1000. # MW to GW
GBaseload = GBaseload / 1000. # MW to GW

if scenario >= 21:
    CostPH, CostDC = -1, -1
else:
    CostPH, CostDC = 0,0 

nyears = 4 

leapdays = (nyears+(4-59/365))//4

ndays = 365*nyears + leapdays



#%%
print("Instantiating model:", dt.now())
model = pyo.ConcreteModel()

model.pvl = pyo.RangeSet(len(PVl))
model.windl = pyo.RangeSet(len(Windl))
model.lines = pyo.RangeSet(len(network))
model.nodes = pyo.RangeSet(nodes)

model.t = pyo.RangeSet(ndays) # first 4 years

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

model.gpv =     pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals, initialize=lambda m, t, n: sum((TSPV[t-1, n-1] * m.cpv[z+1] for z in np.where(PVl_int==Nodel_int[n-1])[0])))    
model.gwind =   pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals, initialize=lambda m, t, n: sum((TSWind[t-1, n-1] * m.cwind[z+1] for z in np.where(Windl_int==Nodel_int[n-1])[0])))

model.eload = pyo.Param(model.t, model.nodes, domain=pyo.Reals, rule=lambda m, t, n: MLoad[t-1, n-1] - GBaseload[t-1, n-1])
model.hvdcCost = pyo.Param(model.lines, domain=pyo.Reals, initialize = dict(zip(range(1, len(network)+1), factor[4:11][network_mask])))

model.charge =  pyo.Var(model.t, model.nodes, domain=pyo.Reals)
model.storage = pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.hvdc =    pyo.Var(model.t, model.lines, domain=pyo.Reals)
model.hydro =   pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.bio =     pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.spillage =pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)

model.constr_gpv =   pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.gpv[t, n]   == sum((TSPV[t-1, n-1] *   m.cpv[z+1]   for z in np.where(PVl_int==  Nodel_int[n-1])[0])))
model.constr_gwind = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.gwind[t, n] == sum((TSWind[t-1, n-1] * m.cwind[z+1] for z in np.where(Windl_int==Nodel_int[n-1])[0])))

model.constr_charge_power_lower = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n:-m.cphp[n] <= m.charge[t, n])
model.constr_charge_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.charge[t, n] <= m.cphp[n])

model.constr_hydro_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.hydro[t, n] <= CHydro[n-1])
model.constr_bio_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.bio[t, n] <= CBio[n-1])

model.constr_storage_energy_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.storage[t, n] <= m.cphs[n])

model.constr_hvdc_lower = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l:-m.chvdc[l] <= m.hvdc[t, l])
model.constr_hvdc_upper = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l: m.hvdc[t, l] <= m.chvdc[l])

def constr_state_of_charge(m, t, n):
    if t==1:
        return m.storage[t, n] == 0.5 * m.cphs[n]
    else:
        return m.storage[t, n] == m.storage[t-1, n] - m.charge[t-1, n] * resolution * efficiency

model.constr_storage_state_of_charge = pyo.Constraint(model.t, model.nodes, rule=constr_state_of_charge)



def constr_power_balance_lower(m, t, n):
    return (m.eload[t, n] - m.gpv[t, n] - m.gwind[t, n] - m.hydro[t,n] - m.bio[t,n] - m.charge[t,n] 
            - sum((m.hvdc[t, l+1] for l in np.where(network[:,0] == n-1)[0])) 
            + sum((m.hvdc[t, l+1] for l in np.where(network[:,0] == n-1)[0])) + m.spillage[t, n] 
            ) >= -0.001
def constr_power_balance_upper(m, t, n):
    return (m.eload[t, n] - m.gpv[t, n] - m.gwind[t, n] - m.hydro[t,n] - m.bio[t,n] - m.charge[t,n] 
            - sum((m.hvdc[t, l+1] for l in np.where(network[:,0] == n-1)[0])) 
            + sum((m.hvdc[t, l+1] for l in np.where(network[:,0] == n-1)[0])) + m.spillage[t, n] 
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
        (factor[13]+0.000_001) * pyo.summation(m.bio)/ 1000. +
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



# model.display()

model.OBJ.display()




