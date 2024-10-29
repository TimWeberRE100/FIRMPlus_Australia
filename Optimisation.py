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
DCloss = DCloss[network_mask]
StartCharge = 0.5
# =============================================================================
# Be careful about Nodel or np.unique(PVl/Windl) when zones don't have a technology
# =============================================================================
pv_zs_in_n = [np.where(PVl==node)[0] + 1 for node in Nodel] 
wind_zs_in_n = [np.where(Windl==node)[0] + 1 for node in Nodel]

import_lines = [np.where(network[:,0]==n)[0] + 1 for n in range(nodes)]
export_lines = [np.where(network[:,1]==n)[0] + 1 for n in range(nodes)]

nhvdc = len(network)
npv = len(PVl)
nwind = len(Windl)

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

model.pvl = pyo.RangeSet(npv)
model.windl = pyo.RangeSet(nwind)
model.lines = pyo.RangeSet(nhvdc)
model.nodes = pyo.RangeSet(nodes)

model.t = pyo.RangeSet(48*ndays) 

model.cpv = pyo.Var(
    model.pvl,   
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, npv+1), zip(npv*[0.], npv*[50.]))),
    initialize=dict(zip(range(1, npv+1), npv*[10.])),
    )
model.cwind = pyo.Var(
    model.windl, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nwind+1), zip(nwind*[0.], nwind*[50.]))),
    initialize=dict(zip(range(1, nwind+1)  , nwind*[10.])),
    )
model.cphp = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(nodes*[0.], nodes*[50.]))),
    initialize=dict(zip(range(1, nodes+1), nodes*[10.])),
    )
model.cphe = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(nodes*[0.], nodes*[500.]))),
    initialize=dict(zip(range(1, nodes+1), nodes*[100.])),
    )
model.chvdc = pyo.Var(
    model.lines, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nhvdc+1), zip(nhvdc*[0.], nhvdc*[100.]))),
    initialize=dict(zip(range(1, nhvdc+1), nhvdc*[50.])),
    )

model.hvdcCost = pyo.Param(model.lines, domain=pyo.Reals, initialize = dict(zip(range(1, nhvdc+1), factor[4:12][network_mask])))

model.charge =  pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.discharge =  pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.storage = pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.hvdc =    pyo.Var(model.t, model.lines, domain=pyo.Reals)
model.hydro =   pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.bio =     pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.spillage =pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)

model.constr_charge_power_lower = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n:-m.cphp[n] <= m.charge[t, n])
model.constr_charge_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.charge[t, n] <= m.cphp[n])

model.constr_discharge_power_lower = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n:-m.cphp[n] <= m.discharge[t, n])
model.constr_discharge_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.discharge[t, n] <= m.cphp[n])

model.constr_storage_energy_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.storage[t, n] <= m.cphe[n])

model.constr_hydro_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.hydro[t, n] <= CHydro[n-1])
model.constr_bio_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.bio[t, n] <= CBio[n-1])

model.constr_hvdc_line_power_lower = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l: m.hvdc[t, l] >= -m.chvdc[l])
model.constr_hvdc_line_power_upper = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l: m.hvdc[t, l] <= m.chvdc[l])
model.constr_import_export_balance = pyo.Constraint(model.t, rule=lambda m, t: sum(m.hvdc[t, l] for l in m.lines) == 0)

model.constr_max_hydro = pyo.Constraint(rule=lambda m: pyo.summation(model.hydro) <= 20_000 * nyears)

def constr_state_of_charge(m, t, n):
    if t==1:
        return m.storage[t, n] == StartCharge * m.cphe[n]
    else:
        return m.storage[t, n] == m.storage[t-1, n] - m.discharge[t-1, n] * resolution + m.charge[t-1, n] * resolution * efficiency

model.constr_storage_state_of_charge = pyo.Constraint(model.t, model.nodes, rule=constr_state_of_charge)


def constr_power_balance(m, t, n):
    return (MLoad[t-1, n-1] + m.spillage[t, n] + m.charge[t,n] 
            - sum((m.cpv[z]*TSPV[t-1, z-1] for z in pv_zs_in_n[n-1])) - sum((m.cwind[z]*TSWind[t-1, z-1] for z in wind_zs_in_n[n-1]))
            - m.hydro[t,n] - m.bio[t,n] - m.discharge[t,n] 
            - sum((m.hvdc[t, l]*(1-DCloss[l-1]) for l in import_lines[n-1])) + sum((m.hvdc[t, l] for l in export_lines[n-1]))
            ) == 0.0

model.constr_power_balance = pyo.Constraint(model.t, model.nodes, rule=constr_power_balance)

def objective(m):
    cost = (
        factor[0] * pyo.summation(m.cpv) +
        factor[1] * pyo.summation(m.cwind) + 
        factor[2] * pyo.summation(m.cphp) +
        factor[3] * pyo.summation(m.cphe) + 
        pyo.summation(m.hvdcCost, m.chvdc) +
        factor[12] * pyo.summation(m.cpv) +
        factor[13] * pyo.summation(m.cwind) + 
        factor[14] * pyo.summation(m.hydro) / 1000. +
        (factor[14]+0.001) * pyo.summation(m.bio) / 1000. + # use hydro first
        factor[15] * CostPH +
        factor[16] * CostDC +

        0)
    LCOE = cost/energy
    #HVDC loss not currently included. Suggest including it in energy balance
    
    # Penalties 
    
    return LCOE #+ Penalties
    

model.OBJ = pyo.Objective(rule=objective)


opt = pyo.SolverFactory('gurobi', tee=True)

start=dt.now()
print("Optimisation starts:", start)
opt.solve(model)
end=dt.now()
print("Optimisation took:", end-start)


#%%
model.OBJ.display()

cpv = np.array([model.cpv[i].value for i in model.cpv])
cwind = np.array([model.cwind[i].value for i in model.cwind])
cphp = np.array([model.cphp[i].value for i in model.cphp])
cphe = np.array([model.cphe[i].value for i in model.cphe])
chvdc = np.array([model.chvdc[i].value for i in model.chvdc])

print('pv:', cpv)
print('wind:', cwind)
print('php:', cphp)
print('phe:', cphe)
print('chvdc:', chvdc)

Charge = np.array([model.charge[i].value for i in model.charge]).reshape(-1, nodes)
Discharge = np.array([model.discharge[i].value for i in model.discharge]).reshape(-1, nodes)
Storage = np.array([model.storage[i].value for i in model.storage]).reshape(-1, nodes)
Hydro = np.array([model.hydro[i].value for i in model.hydro]).reshape(-1, nodes)
Bio = np.array([model.bio[i].value for i in model.bio]).reshape(-1, nodes)
Spillage = np.array([model.spillage[i].value for i in model.spillage]).reshape(-1, nodes)

hvdc = np.array([model.hvdc[i].value for i in model.hvdc]).reshape(-1, nhvdc)

Transmission = np.empty_like(Charge)
for t in range(Charge.shape[0]):
    for n in range(Charge.shape[1]):
        Transmission[t, n] = sum((hvdc[t, l]*(1-DCloss[l-1]) for l in import_lines[n]-1)) - sum((hvdc[t, l] for l in export_lines[n]-1))

#%%
def Debug(length):
    """Debugging"""
    length=int(length)
    PV, Wind = cpv*TSPV[:length, :], cwind*TSWind[:length, :]
    PV = np.stack([PV[:, np.where(PVl==node)[0]].sum(axis=1) for node in Nodel]).T
    Wind = np.stack([Wind[:, np.where(Windl==node)[0]].sum(axis=1) for node in Nodel]).T

    Load = MLoad[:length, :]
    
    for t in range(length):
        #supply-demand
        assert (np.abs(Load[t] + Spillage[t] + Charge[t] - Discharge[t] - Hydro[t] - Bio[t] - Transmission[t] - PV[t] - Wind[t]) <= 0.1).all(), t
    
        # Discharge, Charge and Storage
        if t == 0:
            assert (np.abs(Storage[t] - StartCharge*cphe) <= 0.1).all(), t
        else: 
            assert (np.abs(Storage[t-1] - Storage[t] + Charge[t-1] * resolution * efficiency - Discharge[t-1]* resolution) <= 0.001).all(), t
            
    assert (np.amax(Charge, axis=0)  - cphp <= 0.001).all()
    assert (np.amax(Discharge, axis=0) - cphp <= 0.001).all()
    assert (np.amax(Storage, axis=0) - cphe <= 0.001).all()
    assert (np.amax(hvdc, axis=0) - chvdc <= 0.001).all()
    assert (np.amin(hvdc, axis=0) + chvdc <= 0.001).all()
    
    assert (hvdc.sum(axis=1) <= 0.1).all() #imports = exports at each time


    print('Debugging: everything is ok')

    return True

Debug(ndays*48)
