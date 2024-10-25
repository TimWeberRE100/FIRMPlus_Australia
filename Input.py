# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import float64, int32, types, int64
from numba.experimental import jitclass

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=21, type=int, required=False, help='11, 12, 13, ...')
parser.add_argument('-n', default='Super1', type=str, required=False, help='node=Super1')
parser.add_argument('-w', default=1, type=int, required=False, help='Number of islands in differential evolution (i.e. workers)')
args = parser.parse_args()

scenario = args.s
node = args.n

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))
Nodel_int, PVl_int, Windl_int = (x.astype(np.int64) for x in (Nodel_int, PVl_int, Windl_int))

Nodel_int = Nodel_int.astype(np.int32)
PVl_int = PVl_int.astype(np.int32)
Windl_int = Windl_int.astype(np.int32)

resolution = 0.5

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) # TSWind(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(np.float64)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DCloss = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) * 0.03 * pow(10, -3)

efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

firstyear, finalyear, timestep = (2020, 2029, 1)

network = np.array([[0, 3], #FNQ-QLD
                    [1, 3], #NSW-QLD
                    [1, 4], #NSW-SA
                    [1, 6], #NSW-VIC
                    [2, 4], #NT-SA
                    [4, 7], #SA-WA
                    [5, 6], #TAS-VIC
                    ], dtype=np.int64)
    
if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad = MLoad[:, Nodel==node]
    TSPV = TSPV[:, PVl==node]
    TSWind = TSWind[:, Windl==node]
    CHydro, CBio, CBaseload, CPeak = [x[Nodel==node] for x in (CHydro, CBio, CBaseload, CPeak)]

    Nodel_int, PVl_int, Windl_int = [x[x==n_node[node]] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[x==node] for x in (Nodel, PVl, Windl)]
    network_mask = np.zeros(7, bool)
    network = np.empty((0,0), int)

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

    network_mask = np.array([(network==j).sum(axis=1).astype(np.bool_) for j in Nodel_int]).sum(axis=0)==2
    network = network[network_mask,:]
    networkdict = {v:k for k, v in enumerate(Nodel_int)}
    #translate into indicies rather than Nodel_int values
    network = np.array([networkdict[n] for n in network.flatten()], np.int64).reshape(network.shape)


if scenario >= 31:
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    
    TSPV = np.stack([TSPV[:, PVl==node].mean(axis=1) for node in coverage]).T
    TSWind = np.stack([TSWind[:, Windl==node].mean(axis=1) for node in coverage]).T
    # having full of zeros and setting lb,ub=0,0 makes code faster
    TSPV = np.nan_to_num(TSPV, False, 0)
    warnings.simplefilter('default', RuntimeWarning)
    
    Nodel_int, PVl_int, Windl_int = [np.unique(x) for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [np.unique(x)  for x in (Nodel, PVl, Windl)]
    
intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

# Specify the types for jitclass
solution_spec = [
    ('x', float64[:]),  # Assuming x is a list of floats
    ('MLoad', float64[:, :]),  # 2D array of floats
    ('intervals', int32),
    ('nodes', int32),
    ('resolution',float64),
    ('CPV', float64[:]),
    ('CWind', float64[:]),
    ('GPV', float64[:, :]),  # 2D array of floats
    ('GWind', float64[:, :]),  # 2D array of floats
    ('CPHP', float64[:]),
    ('CPHS', float64),
    ('efficiency', float64),
    ('CInter', float64[:]),
    ('GInter', float64[:, :]),  # 2D array of floats
    ('Nodel_int', int32[:]), 
    ('PVl_int', int32[:]),
    ('Windl_int', int32[:]),
    ('Interl_int', int32[:]),
    ('node', types.unicode_type),
    ('GBaseload', float64[:, :]),  # 2D array of floats
    ('CPeak', float64[:]),  # 1D array of floats
    ('CHydro', float64[:]),  # 1D array of floats
    ('EHydro', float64[:]),  # 1D array of floats
    ('allowance', float64),
    ('flexible', float64[:,:]),
    ('Discharge', float64[:,:]),
    ('Charge', float64[:,:]),
    ('Storage', float64[:,:]),
    ('Deficit', float64[:,:]),
    ('Spillage', float64[:,:])
]

@jitclass(solution_spec)
class Solution:
    #A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)
    
    def __init__(self, x):
        self.x = x
        self.MLoad = MLoad
        self.intervals = intervals
        self.nodes = nodes
        self.resolution = resolution

        self.CPV = x[: pidx]  # CPV(i), GW
        self.CWind = x[pidx: widx]  # CWind(i), GW
        """ if node == 'Super2':
            self.CInter = NumbaList(x[sidx+1: iidx]) # CInter(j), GW
        else:
            self.CInter = NumbaList([0.0])  # CInter(j), GW """
        
        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, len(self.CPV)))
        CWind_tiled = np.zeros((intervals, len(self.CWind)))
        #CInter_tiled = np.zeros((intervals, len(self.CWind)))
        for i in range(intervals):
            for j in range(len(self.CPV)):
                CPV_tiled[i, j] = self.CPV[j]
            for j in range(len(self.CWind)):
                CWind_tiled[i, j] = self.CWind[j]
            """ for j in range(len(self.CInter)):
                CInter_tiled[i, j] = self.CInter[j] """

        self.GPV = TSPV * CPV_tiled * 1000  # GPV(i, t), GW to MW
        self.GWind = TSWind * CWind_tiled * 1000  # GWind(i, t), GW to MW
        #self.GInter = CWind_tiled * 1000  # GInter(j, t), GW to MW

        self.CPHP = x[widx: sidx]  # CPHP(j), GW
        self.CPHS = x[sidx]  # S-CPHS(j), GWh
        self.efficiency = efficiency

        self.Nodel_int = Nodel_int
        self.PVl_int = PVl_int
        self.Windl_int = Windl_int
        #self.Interl = NumbaList([str(item) for item in Interl])
        self.node = node

        self.GBaseload = GBaseload
        self.CPeak = CPeak
        self.CHydro = CHydro
        #self.EHydro = EHydro


from Simulation import Reliability
from Network import Transmission

def F(x):
    """This is the objective function."""

    S = Solution(x)

    Deficit = Reliability(S, flexible=np.zeros(intervals, dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum() * resolution / years / efficiency # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = max(0, Hydro - 20 * 1000000) # TWh p.a. to MWh p.a.

    TDC = Transmission(S) if 'Super' in node else np.zeros((intervals, len(DCloss)), dtype=np.float64)  # TDC: TDC(t, k), MW
    TDC_abs = np.abs(TDC)

    Deficit = Reliability(S, flexible=np.ones(intervals, dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
    Deficit_sum = Deficit.sum() * resolution
    PenDeficit = max(0, Deficit_sum) # MWh

    CDC = np.zeros(len(DCloss), dtype=np.float64)
    for i in range(0,intervals):
        for j in range(0,len(DCloss)):
            if TDC_abs[i][j] > CDC[j]:
                CDC[j] = TDC_abs[i][j]
    CDC = CDC * 0.001 # CDC(k), MW to GW

    cost = factor *  np.concatenate((np.array([S.CPV.sum(), S.CWind.sum(), S.CPHP.sum(), S.CPHS]), CDC, np.array([S.CPV.sum(), S.CWind.sum(), Hydro * 0.000001, -1.0, -1.0])))
    cost = cost.sum()

    loss = TDC_abs.sum(axis=0) * DCloss
    loss = loss.sum() * 0.000000001 * resolution / years # PWh p.a.
    LCOE = cost / abs(energy - loss)

    Func = LCOE + PenDeficit + PenHydro

    return Func
