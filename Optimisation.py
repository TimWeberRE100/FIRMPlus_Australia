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

masked_DCloss = DCloss[network_mask]

pv_zs_in_n = [np.where(PVl==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing
wind_zs_in_n = [np.where(Windl==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing

pos_export_lines = [np.where(network[:,0]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing
neg_export_lines = [np.where(network[:,1]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing

npv = len(PVl)
nwind = len(Windl)

if scenario >= 21:
    LegPH, LegINTC = -1, -1
else:
    LegPH, LegINTC = 0,0 

leapdays = (years+(4-59/365))//4

ndays = 365*years + leapdays
intervals = int(ndays*24/resolution)

#%%
print("Instantiating optimiser:", dt.now())
model = pyo.ConcreteModel()

adj_energy = (MLoad[:intervals, :].sum() * pow(10, -6) * resolution / years)

model.pvl = pyo.RangeSet(npv)
model.windl = pyo.RangeSet(nwind)
model.lines = pyo.RangeSet(nhvdc)
model.nodes = pyo.RangeSet(nodes)

model.t = pyo.RangeSet(intervals) 

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
    bounds=dict(zip(range(1, nodes+1), zip(contingency, nodes*[50.]))),
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
model.discharge=pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.storage = pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.hvdc_pos = pyo.Var(model.t, model.lines, domain=pyo.NonNegativeReals)
model.hvdc_neg = pyo.Var(model.t, model.lines, domain=pyo.NonNegativeReals)
model.hydro =   pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)
model.bio =     pyo.Var(model.t, model.nodes, domain=pyo.NonNegativeReals)

model.constr_charge_power_lower = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n:-m.cphp[n] <= m.charge[t, n])
model.constr_charge_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.charge[t, n] <= m.cphp[n])

model.constr_discharge_power_lower = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n:-m.cphp[n] <= m.discharge[t, n])
model.constr_discharge_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.discharge[t, n] <= m.cphp[n])

model.constr_storage_energy_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.storage[t, n] <= m.cphe[n])

model.constr_hydro_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.hydro[t, n] <= CHydro[n-1])
model.constr_bio_power_upper = pyo.Constraint(model.t, model.nodes, rule=lambda m, t, n: m.bio[t, n] <= CBio[n-1])

model.constr_hvdc_power_import = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l: m.hvdc_pos[t, l] <= m.chvdc[l])
model.constr_hvdc_power_export = pyo.Constraint(model.t, model.lines, rule=lambda m, t, l: m.hvdc_neg[t, l] <= m.chvdc[l])


model.constr_max_hydrobio = pyo.Constraint(rule=lambda m: pyo.summation(m.hydro)*0.001*resolution/years
                                           + pyo.summation(m.bio)*0.001*resolution/years <= 20.0) #TWh p.a.

def constr_state_of_charge(m, t, n):
    if t==1:
        return m.storage[t, n] == StartCharge * m.cphe[n]
    else:
        return m.storage[t, n] == m.storage[t-1, n] - m.discharge[t-1, n] * resolution + m.charge[t-1, n] * resolution * efficiency

model.constr_storage_state_of_charge = pyo.Constraint(model.t, model.nodes, rule=constr_state_of_charge)

    
def constr_power_balance(m, t, n):
    return (MLoad[t-1, n-1] + m.charge[t,n] 
            - sum((m.cpv[z]*TSPV[t-1, z-1] for z in pv_zs_in_n[n-1])) - sum((m.cwind[z]*TSWind[t-1, z-1] for z in wind_zs_in_n[n-1]))
            - m.hydro[t,n] - m.bio[t,n] - m.discharge[t,n] 
            + sum((m.hvdc_pos[t, l] - m.hvdc_neg[t, l]*(1-masked_DCloss[l-1]) for l in pos_export_lines[n-1]))
            + sum((m.hvdc_neg[t, l] - m.hvdc_pos[t, l]*(1-masked_DCloss[l-1]) for l in neg_export_lines[n-1]))
            ) <= 0.0

model.constr_power_balance = pyo.Constraint(model.t, model.nodes, rule=constr_power_balance)

model.CostPV =    pyo.Expression(rule=lambda m: factor[0] *pyo.summation(m.cpv)  )
model.CostWind =  pyo.Expression(rule=lambda m: factor[1] *pyo.summation(m.cwind))
model.CostPH =    pyo.Expression(rule=lambda m: factor[2] *pyo.summation(m.cphp)  +
                                                factor[3] *pyo.summation(m.cphe)  +
                                                factor[15]*LegPH)
model.CostHydro = pyo.Expression(rule=lambda m: factor[14]*pyo.summation(m.hydro)*0.001*resolution/years)
model.CostBio =   pyo.Expression(rule=lambda m: factor[14]*pyo.summation(m.bio)*0.001*resolution/years)
model.CostDC =    pyo.Expression(rule=lambda m: pyo.summation(m.hvdcCost, m.chvdc) + 
                                                factor[16]*LegINTC)
model.CostAC =    pyo.Expression(rule=lambda m: factor[12]*pyo.summation(m.cpv)   +
                                                factor[13]*pyo.summation(m.cwind))

def objective(m):
    return (m.CostPV + m.CostWind + m.CostPH + m.CostHydro + m.CostBio +
            m.CostDC + m.CostAC) / adj_energy
    
model.OBJ = pyo.Objective(rule=objective)

optimiser = pyo.SolverFactory('gurobi')

start=dt.now()
print("Optimisation starts:", start)
optimiser.solve(model)
end=dt.now()
print("Optimisation took:", end-start)

model.OBJ.display()

#%%

S = Solution(model, years)

print('pv:', S.cpv)
print('wind:', S.cwind)
print('php:', S.cphp)
print('phe:', S.cphe)
print('chvdc:', S.chvdc)

try:
    with open(f'Results/Optimisation_resultx{scenario}.csv', 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([S.OBJ] + list(S.cpv) + list(S.cwind) + list(S.cphp) + list(S.cphe) + list(S.chvdc))
except FileNotFoundError:
    import os 
    os.mkdir('Results')
    with open(f'Results/Optimisation_resultx{scenario}.csv', 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([S.OBJ] + list(S.cpv) + list(S.cwind) + list(S.cphp) + list(S.cphe) + list(S.chvdc))

from Statistics import Information
Information(S)
