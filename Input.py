# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit, float64, int64, prange, boolean
from numba.experimental import jitclass

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')

parser.add_argument('-s', default=21, type=int, required=False, help='11, 12, 13, ...')

parser.add_argument('-cb', default=2, type=int, required=False, help='Callback: 0-None, 1-generation elites, 2-everything')
parser.add_argument('-ver', default=1, type=int, required=False, help='Boolean - print progress to console')
parser.add_argument('-vp', default=50, type=int, required=False, help='Maximum number of vectors to send to objective')
parser.add_argument('-w', default=1, type=int, required=False, help='Maximum number of cores to parallelise over')
parser.add_argument('-vec', default=0, type=int, required=False, help='Boolean - vectorised mode')
parser.add_argument('-resume', default=0, type=int, required=False, help='Boolean - whether to restart')

args = parser.parse_args()
assert args.w > 0 or args.w in (-1, -2)
scenario = args.s

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))

Nodel_int, PVl_int, Windl_int = (x.astype(np.int64) for x in (Nodel_int, PVl_int, Windl_int))

resolution = 0.5
firstyear, finalyear, timestep = (2020, 2029, 1)

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) # TSWind(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DCloss = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) * 0.03 * pow(10, -3)
CDC6max = 3 * 0.63 # GW

efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad = MLoad[:, Nodel==node]
    TSPV = TSPV[:, PVl==node]
    TSWind = TSWind[:, Windl==node]
    CHydro, CBio, CBaseload, CPeak = [x[Nodel==node] for x in (CHydro, CBio, CBaseload, CPeak)]

    Nodel_int, PVl_int, Windl_int = [x[x==n_node[node]] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[x==node] for x in (Nodel, PVl, Windl)]

elif scenario>=21:
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1] 
    
    MLoad = MLoad[:, np.in1d(Nodel, coverage)]
    TSPV = TSPV[:, np.in1d(PVl, coverage)]
    TSWind = TSWind[:, np.in1d(Windl, coverage)]
    CHydro, CBio, CBaseload, CPeak = [x[np.in1d(Nodel, coverage)] for x in (CHydro, CBio, CBaseload, CPeak)]
    
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        
    coverage_int = np.array([n_node[node] for node in coverage])
    Nodel_int, PVl_int, Windl_int = [x[np.isin(x, coverage_int)] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[np.isin(x, coverage)] for x in (Nodel, PVl, Windl)]

    
intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW



lb = np.array([0.]  * pzones + [0.]   * wzones + contingency   + [0.])
# ub = np.array([16.] * pzones + [16.]  * wzones + list(np.array(contingency)+16) + [512.])
ub = np.array([50.] * pzones + [50.]  * wzones + nodes*[50.] + [2048.])

#%%
# from Simulation import Reliability, VReliability
from Network import Transmission, VTransmission

# from Simulation import  VReliability
# from Sim2 import Reliability

from Sim2 import Reliability, VReliability


@njit()
def vF(S):
    assert S.vectorised is True
    nvec = S.nvec
    
    Deficit = VReliability(S, flexible=np.zeros((intervals, 1) , dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum(axis=0) * resolution / years / efficiency # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = np.maximum(0, Hydro - 20 * 1000000) # TWh p.a. to MWh p.a.

    Deficit = VReliability(S, flexible=np.ones((intervals, 1), dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
    PenDeficit = np.maximum(0, Deficit.sum(axis=0) * resolution) # MWh

    TDC_abs = VTransmission(S) if scenario>=21 else np.zeros((intervals, len(DCloss)), dtype=np.float64)  # TDC: TDC(t, k), MW
    TDC_abs = np.atleast_3d(np.abs(TDC_abs)).transpose(1,0,2)

    CDC = np.zeros((len(DCloss), nvec), dtype=np.float64)
    for j in prange(len(DCloss)):
        for i in range(intervals):
            CDC[j, :] = np.maximum(TDC_abs[j, i, :], CDC[j,:])
    CDC = CDC * 0.001 # CDC(k), MW to GW
    PenDC = np.maximum(0, CDC[6,:] - CDC6max) * 0.001 # GW to MW

    _c = 0 if scenario <= 17 else -1
    # numba is fussy about generation of tuples and about stacking arrays of different dimensions
    costitems = np.vstack((S.CPV.sum(axis=0), S.CWind.sum(axis=0), S.CPHP.sum(axis=0), S.CPHS,
                            S.CPV.sum(axis=0), S.CWind.sum(axis=0), Hydro * 0.000001,
                            np.repeat(_c, nvec), np.repeat(_c, nvec),))
    costitems = np.vstack((costitems, CDC))
    reindex = np.concatenate((np.arange(4), np.arange(9, 16), np.arange(4, 9)))

    cost = factor.reshape(-1,1) * costitems[reindex]
    cost = cost.sum(axis=0)

    loss = TDC_abs.sum(axis=1) * DCloss.reshape(-1,1)
    loss = loss.sum(axis=0) * 0.000000001 * resolution / years # PWh p.a.
    LCOE = cost / np.abs(energy - loss)
    
    return LCOE, (PenHydro+PenDeficit+PenDC)


# Specify the types for jitclass
vsolution_spec = [
    ('x', float64[:,:]),  # x is 2d array
    ('nvec', int64),
    ('MLoad', float64[:, :, :]),  # 3D array of floats
    ('intervals', int64),
    ('nodes', int64),
    ('resolution',float64),
    ('CPV', float64[:, :]), # 2D array of floats
    ('CWind', float64[:, :]), # 2D array of floats
    ('GPV', float64[:, :, :]),  # 3D array of floats
    ('GWind', float64[:, :, :]),  # 3D array of floats
    ('CPHP', float64[:, :]),
    ('CPHS', float64[:]),
    ('efficiency', float64),
    ('Nodel_int', int64[:]), 
    ('PVl_int', int64[:]),
    ('Windl_int', int64[:]),
    ('GBaseload', float64[:, :, :]),  # 3D array of floats
    ('CPeak', float64[:]),  # 1D array of floats
    ('CHydro', float64[:]),  # 1D array of floats
    ('flexible', float64[:,:]),
    ('Discharge', float64[:,:]),
    ('Charge', float64[:,:]),
    ('Storage', float64[:,:]),
    ('Deficit', float64[:,:]),
    ('Spillage', float64[:,:]),
    ('Penalties', float64[:]),
    ('Lcoe', float64[:]),
    ('evaluated', boolean),
    ('vectorised',boolean),
    ('MPV', float64[:, :, :]),
    ('MWind', float64[:, :, :]),
    ('MBaseload', float64[:, :, :]),
    ('MPeak', float64[:, :, :]),
    ('MDischarge', float64[:, :, :]),
    ('MCharge', float64[:, :, :]),
    ('MStorage', float64[:, :, :]),
    ('MDeficit', float64[:, :, :]),
    ('MSpillage', float64[:, :, :]),
]

@jitclass(vsolution_spec)
class VSolution:
    #A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)
    
    def __init__(self, x):
        # input vector should have shape (sidx+1, n) i.e. vertical input vectors
        self.vectorised=True
        assert x.shape[0] == len(lb)
        
        self.x = x
        self.nvec = x.shape[1]
        
        self.intervals, self.nodes = intervals, nodes
        self.resolution = resolution
       
        shape3d = (intervals, nodes, 1)
        self.MLoad = MLoad.reshape(shape3d)

        self.CPV = x[: pidx, :]  # CPV(i), GW
        self.CWind = x[pidx: widx, :]  # CWind(i), GW
        
        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, *self.CPV.shape))
        CWind_tiled = np.zeros((intervals, *self.CWind.shape))
        # CInter_tiled = np.zeros((intervals, len(self.CWind)))
        for i in range(intervals):
            for j in prange(len(self.CPV)):
                CPV_tiled[i, j, :] = self.CPV[j, :]
            for j in prange(len(self.CWind)):
                CWind_tiled[i, j, :] = self.CWind[j, :]
                
        self.GPV = TSPV.reshape((*TSPV.shape, 1)) * CPV_tiled * 1000.  # GPV(i, t), GW to MW
        self.GWind = TSWind.reshape((*TSWind.shape, 1)) * CWind_tiled * 1000.  # GWind(i, t), GW to MW

        self.CPHP = x[widx: sidx, :]  # CPHP(j), GW
        self.CPHS = x[sidx, :]  # S-CPHS(j), GWh
        self.efficiency = efficiency

        self.Nodel_int, self.PVl_int, self.Windl_int = Nodel_int, PVl_int, Windl_int

        self.GBaseload = GBaseload.reshape(shape3d)
        self.CPeak = CPeak
        self.CHydro = CHydro
        self.evaluated=False

    # @staticmethod
    def _evaluate(self):
        self.Lcoe, self.Penalties = vF(self)
        self.evaluated=True
        
    # # Not currently supported by jitclass
    # def __repr__(self):
    #     """S = Solution(list(np.ones(64))) >> print(S)"""
    #     return 'Solution({})'.format(self.x)

#%%
@njit()
def F(S):
    assert S.vectorised is False
    
    Deficit = Reliability(S, flexible=np.zeros((intervals, ) , dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum(axis=0) * resolution / years / efficiency # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = np.maximum(0, Hydro - 20_000_000) # TWh p.a. to MWh p.a.

    Deficit = Reliability(S, flexible=np.ones((intervals, ), dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
    PenDeficit = np.maximum(0, Deficit.sum(axis=0) * resolution) # MWh

    TDC_abs = np.abs(Transmission(S)) if scenario>=21 else np.zeros((intervals, len(DCloss)), dtype=np.float64)  # TDC: TDC(t, k), MW

    CDC = np.zeros(len(DCloss), dtype=np.float64)
    for j in prange(len(DCloss)):
        for i in range(intervals):
            CDC[j] = np.maximum(TDC_abs[i, j], CDC[j])
    CDC = CDC * 0.001 # CDC(k), MW to GW
    PenDC = max(0, CDC[6] - CDC6max) * 0.001 # GW to MW

    _c = 0 if scenario <= 17 else -1
    cost = (factor * np.array([S.CPV.sum(), S.CWind.sum(), S.CPHP.sum(), S.CPHS] + list(CDC) +
                              [S.CPV.sum(), S.CWind.sum(), Hydro * 0.000_001, _c, _c])
            )

    loss = TDC_abs.sum(axis=0) * DCloss
    loss = loss.sum(axis=0) * 0.000_000_001 * resolution / years # PWh p.a.
    energyloss = np.abs(energy - loss)
    LCOE = cost.sum() / energyloss
    LCOG = 1000 * cost[np.array([0, 1, 13])].sum() / (
        0.000_001*(resolution/years*(S.GPV.sum() + S.GWind.sum()) + Hydro))
    LCOBS = cost[np.array([2,3,14])].sum()/energyloss
    LCOBT = cost[np.array([4,5,6,7,8,9,10,11,12,15])].sum()/energyloss
    LCOBL = LCOE - LCOG - LCOBS - LCOBT
    
    return LCOE, (PenHydro+PenDeficit+PenDC), LCOG, LCOBS, LCOBT, LCOBL

# Specify the types for jitclass
solution_spec = [
    ('x', float64[:]),  # x is 1d array
    ('nvec', int64),
    ('MLoad', float64[:, :]),  # 2D array of floats
    ('intervals', int64),
    ('nodes', int64),
    ('resolution',float64),
    ('CPV', float64[:]), # 1D array of floats
    ('CWind', float64[:]), # 1D array of floats
    ('GPV', float64[:, :]),  # 2D array of floats
    ('GWind', float64[:, :]),  # 2D array of floats
    ('CPHP', float64[:,]),
    ('CPHS', float64),
    ('efficiency', float64),
    ('Nodel_int', int64[:]), 
    ('PVl_int', int64[:]),
    ('Windl_int', int64[:]),
    ('GBaseload', float64[:, :]),  # 2D array of floats
    ('CPeak', float64[:]),  # 1D array of floats
    ('CHydro', float64[:]),  # 1D array of floats
    ('flexible', float64[:]),
    ('Discharge', float64[:]),
    ('Charge', float64[:]),
    ('Storage', float64[:]),
    ('Deficit', float64[:]),
    ('Spillage', float64[:]),
    ('Netload' ,float64[:]),
    ('Penalties', float64),
    ('LCOE', float64),
    ('LCOG', float64),
    ('LCOBS', float64),
    ('LCOBT', float64),
    ('LCOBL', float64),
    ('evaluated', boolean),
    ('vectorised',boolean),
    ('MPV', float64[:, :]),
    ('MWind', float64[:, :]),
    ('MBaseload', float64[:, :]),
    ('MPeak', float64[:, :]),
    ('MDischarge', float64[:, :]),
    ('MCharge', float64[:, :]),
    ('MStorage', float64[:, :]),
    ('MDeficit', float64[:, :]),
    ('MSpillage', float64[:, :]),
    ('MHydro', float64[:, :]),
    ('MBio', float64[:, :]),
    ('TDC', float64[:, :]),
    ('CDC', float64[:]),
    ('FQ', float64[:]),
    ('NQ', float64[:]),
    ('NS', float64[:]),
    ('NV', float64[:]),
    ('AS', float64[:]),
    ('SW', float64[:]),
    ('TV', float64[:]),
    ('Topology', float64[:, :]),
]

@jitclass(solution_spec)
class Solution:
    #A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)
    
    def __init__(self, x):
        # input vector should have shape (sidx+1, n) i.e. vertical input vectors
        self.vectorised=False
        assert len(x) == len(lb)
        
        self.x = x
        self.nvec = 1
        
        self.intervals, self.nodes = intervals, nodes
        self.resolution = resolution
       
        self.MLoad = MLoad

        self.CPV = x[: pidx]  # CPV(i), GW
        self.CWind = x[pidx: widx]  # CWind(i), GW

        self.GPV = TSPV * np.ones((intervals, len(self.CPV))) * self.CPV * 1000.  # GPV(i, t), GW to MW
        self.GWind = TSWind * np.ones((intervals, len(self.CWind))) * self.CWind * 1000.  # GWind(i, t), GW to MW

        self.CPHP = x[widx: sidx]  # CPHP(j), GW
        self.CPHS = x[sidx]  # S-CPHS(j), GWh
        self.efficiency = efficiency

        self.Nodel_int, self.PVl_int, self.Windl_int = Nodel_int, PVl_int, Windl_int
        
        self.GBaseload = GBaseload
        self.CPeak = CPeak
        self.CHydro = CHydro
        
        self.evaluated=False
        
    def _evaluate(self):
        self.LCOE, self.Penalties, self.LCOG, self.LCOBS, self.LCOBT, self.LCOBL = F(self)
        self.evaluated=True

    # def __repr__(self):
    #     """S = Solution(list(np.ones(64))) >> print(S)"""
    #     return 'Solution({})'.format(self.x)

if __name__=='__main__':
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    solution = Solution(x)#/1.25) 
    solution._evaluate()
    print(solution.LCOE, solution.Penalties)
    print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)

    
    def test():
        x = np.random.rand(len(lb))*(ub-lb)+lb
        solution = Solution(x)#/1.25) 
        solution._evaluate()
        print(solution.LCOE, solution.Penalties)
        print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)
    # test()
        
    def testV(n):
        x = (np.random.rand(n, len(lb))*np.atleast_2d(ub-lb)+lb).T
        solution = VSolution(x)#/1.25) 
        solution._evaluate()
        print(solution.Lcoe, solution.Penalties)
    # testV(2)
    