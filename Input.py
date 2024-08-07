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
parser.add_argument('-cb', default=0, type=int, required=False, help='Callback: 0-None, 1-generation elites, 2-everything')
parser.add_argument('-ver', default=1, type=int, required=False, help='Boolean - print progress to console')
parser.add_argument('-w', default=-1, type=int, required=False, help='Maximum number of cores to parallelise over')

args = parser.parse_args()
assert args.w > 0 or args.w in (-1, -2)
scenario = args.s

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))

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
CDCmax = [100.]*7
CDCmax[6] = 3 * 0.63 # GW

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
    network = np.empty((0,0,0,0), dtype=np.int64)
    network_mask = np.zeros(7, dtype=np.bool_)
    directconns = np.empty((0,0), dtype=np.int64) 
    trans_tdc_mask = np.empty((0,0), dtype=np.bool_)

elif scenario>=21:
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1] 
    
    network = np.array([[0, 3], #FNQ-QLD
                        [1, 3], #NSW-QLD
                        [1, 4], #NSW-SA
                        [1, 6], #NSW-VIC
                        [2, 4], #NT-SA
                        [4, 7], #SA-WA
                        [5, 6], #TAS-VIC
                        ], dtype=np.int64)
    
    MLoad = MLoad[:, np.isin(Nodel, coverage)]
    TSPV = TSPV[:, np.isin(PVl, coverage)]
    TSWind = TSWind[:, np.isin(Windl, coverage)]
    CHydro, CBio, CBaseload, CPeak = [x[np.isin(Nodel, coverage)] for x in (CHydro, CBio, CBaseload, CPeak)]
    
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
    
    coverage_int = np.array([n_node[node] for node in coverage])
    
    Nodel_int, PVl_int, Windl_int = [x[np.isin(x, coverage_int)] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[np.isin(x, coverage)] for x in (Nodel, PVl, Windl)]
    
    #direct network connections
    network_mask = np.array([(network==j).sum(axis=1).astype(np.bool_) for j in Nodel_int]).sum(axis=0)==2
    network = network[network_mask,:]
    networkdict = {v:k for k, v in enumerate(Nodel_int)}
    #translate into indicies rather than Nodel_int values
    network = np.array([networkdict[n] for n in network.flatten()], np.int64).reshape(network.shape)
    
    trans_tdc_mask = np.zeros((MLoad.shape[1], len(network)), np.bool_)
    for line, row in enumerate(network):
        trans_tdc_mask[row[0], line] = True
    
    directconns = -1*np.ones((len(Nodel)+1, len(Nodel)+1), np.int64)
    for n, row in enumerate(network):
        directconns[*row] = n
        directconns[*row[::-1]] = n
    
    # nodeline_mask = np.zeros((len(network), MLoad.shape[1]), np.bool_)
    # for i in range(nodeline_mask.shape[0]):
    #     for j in range(nodeline_mask.shape[1]):
    #         nodeline_mask[i,j] = i in directconns[j]
    
    def network_neighbours(n):
        isn_mask = np.isin(network, n)
        hasn_mask = isn_mask.sum(axis=1).astype(bool)
        joins_n = network[hasn_mask][~isn_mask[hasn_mask]]
        return joins_n
    
    def nthary_network(network_1):
        """primary, secondary, tertiary, ..., nthary"""
        """supply n-1thary to generate nthary etc."""
        networkn = -1*np.ones((1,network_1.shape[1]+1), dtype=np.int64)
        for row in network_1:
            _networkn = -1*np.ones((1,network_1.shape[1]+1), dtype=np.int64)
            joins_start = network_neighbours(row[0])
            joins_end = network_neighbours(row[-1])
            for n in joins_start:
                if n not in row:
                    _networkn = np.vstack((_networkn, np.insert(row, 0, n)))
            for n in joins_end:
                if n not in row:
                    _networkn = np.vstack((_networkn, np.append(row, n)))
            _networkn=_networkn[1:,:]
            dup=[]
            # find rows which are already in network
            for i, r in enumerate(_networkn): 
                for s in networkn:
                    if np.setdiff1d(r, s).size==0:
                        dup.append(i)
            # find duplicated rows within n3
            for i, r in enumerate(_networkn):
                for j, s in enumerate(_networkn):
                    if i==j:
                        continue
                    if np.setdiff1d(r, s).size==0:
                        dup.append(i)
            _networkn = np.delete(_networkn, np.unique(np.array(dup, dtype=np.int64)), axis=0)
            if _networkn.size>0:
                networkn = np.vstack((networkn, _networkn))
        networkn = networkn[1:,:]
        return networkn
    
    #This version of FIRM maxes out at quarternary transmission
    networks = [network]
    while True:
        n = nthary_network(networks[-1])
        if n.size > 0:
            networks.append(n)
        else: 
            break

    def count_lines(network):
        unique, counts = np.unique(network[:, np.array([0,-1])], return_counts=True)
        if counts.size > 0:
            return counts.max()
        return 0
    maxconnections = max([count_lines(network) for network in networks])

    perfect = np.array([0,1,3,6,10,15,21]) #that's more than enough for now

    network = -1*np.ones((2, len(Nodel), perfect[len(networks)], maxconnections), dtype=np.int64)
    for i, net in enumerate(networks):
        conns = np.zeros(len(Nodel), int)
        for j, row in enumerate(net):
            network[0, row[0], perfect[i]:perfect[i+1], conns[row[0]]] = row[1:]
            network[0, row[-1], perfect[i]:perfect[i+1], conns[row[-1]]] = row[:-1][::-1]
            conns[row[0]]+=1
            conns[row[-1]]+=1
            
    for i in range(network.shape[1]):
        for j in range(network.shape[2]):
            for k in range(network.shape[3]):
                if j in perfect:
                    start=i
                else: 
                    start=network[0, i, j-1, k]
                network[1, i, j, k] = directconns[start, network[0, i, j, k]]

    directconns=directconns[:-1, :-1]
    
        
intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx = pzones, pzones + wzones
spidx, seidx = pzones + wzones + nodes, pzones + wzones + nodes + nodes

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

lb = np.array([0.]  * pzones + [0.]   * wzones + contingency   + [0.] * nodes   + [0.] * len(networks[0]))
ub = np.array([50.] * pzones + [50.]  * wzones + [50.] * nodes + [500.] * nodes + list(np.array(CDCmax)[network_mask]))

#%%
from Simulation import Reliability

@njit()
def F(S):
    assert S.vectorised is False
    
    Deficit = Reliability(S, flexible=np.zeros((intervals, nodes) , dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum() * resolution / years / efficiency # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = np.maximum(0, Hydro - 20 * 1000000) # TWh p.a. to MWh p.a.

    Deficit = Reliability(S, flexible=np.ones((intervals, nodes), dtype=np.float64)*CPeak*1000) # Sj-EDE(t, j), GW to MW
    PenDeficit = np.maximum(0, Deficit.sum() * resolution) # MWh

    CHVDC = np.zeros(len(network_mask), dtype=np.float64)
    CHVDC[network_mask] = S.CHVDC

    _c = 0.0 if scenario <= 17 else -1.0
    cost = (factor * np.array([S.CPV.sum(), S.CWind.sum(), S.CPHP.sum(), S.CPHS.sum()] + list(CHVDC) +
                               [S.CPV.sum(), S.CWind.sum(), Hydro * 0.000001, _c, _c])
            ).sum()

    loss = np.zeros(len(network_mask), dtype=np.float64)
    loss[network_mask] = S.TDC.sum(axis=0) * DCloss[network_mask]
    loss = loss.sum() * 0.000000001 * resolution / years # PWh p.a.
    LCOE = cost / np.abs(energy - loss)
    
    return LCOE, (PenHydro+PenDeficit)

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
    ('CPHP', float64[:]),
    ('CPHS', float64[:]),
    ('CHVDC', float64[:]),
    ('efficiency', float64),
    # ('Nodel_int', int64[:]), 
    # ('PVl_int', int64[:]),
    # ('Windl_int', int64[:]),
    ('GBaseload', float64[:, :]),  # 2D array of floats
    ('CPeak', float64[:]),  # 1D array of floats
    ('CHydro', float64[:]),  # 1D array of floats
    ('flexible', float64[:,:]),
    ('Discharge', float64[:,:]),
    ('Charge', float64[:,:]),
    ('Storage', float64[:,:]),
    ('Deficit', float64[:,:]),
    ('Spillage', float64[:,:]),
    ('Netload' ,float64[:,:]),
    ('Import', float64[:, :]),
    ('Export', float64[:, :]),
    ('Penalties', float64),
    ('Lcoe', float64),
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
    ('CDP', float64[:]),
    ('CDS', float64[:]),
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
    ('network', int64[:, :, :, :]),
    ('directconns', int64[:,:]),
    ('trans_tdc_mask', boolean[:,:]),

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
        self.network, self.directconns = network, directconns
        self.trans_tdc_mask = trans_tdc_mask
        
        # self.network2, self.network3, self.network4 = network2, network3, network4
       
        self.MLoad = MLoad

        self.CPV = x[: pidx]  # CPV(i), GW
        self.CWind = x[pidx: widx]  # CWind(i), GW
        
        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, len(self.CPV)))
        CWind_tiled = np.zeros((intervals, len(self.CWind)))
        # CInter_tiled = np.zeros((intervals, len(self.CWind)))
        for i in range(intervals):
            for j in range(len(self.CPV)):
                CPV_tiled[i, j] = self.CPV[j]
            for j in range(len(self.CWind)):
                CWind_tiled[i, j] = self.CWind[j]

        GPV = TSPV * CPV_tiled * 1000.  # GPV(i, t), GW to MW
        GWind = TSWind * CWind_tiled * 1000.  # GWind(i, t), GW to MW
        
        self.GPV, self.GWind = np.empty((intervals, nodes), np.float64), np.empty((intervals, nodes), np.float64)
        for i, j in enumerate(Nodel_int):
            self.GPV[:,i] = GPV[:, PVl_int==j].sum(axis=1)
            self.GWind[:,i] = GWind[:, Windl_int==j].sum(axis=1) 
            
        self.CPHP = x[widx: spidx]  # CPHP(j), GW
        self.CPHS = x[spidx: seidx]  # S-CPHS(j), GWh
        self.CHVDC = x[seidx:]
        
        self.efficiency = efficiency

        # self.Nodel_int, self.PVl_int, self.Windl_int = Nodel_int, PVl_int, Windl_int
        

        self.GBaseload = GBaseload
        self.CPeak = CPeak
        self.CHydro = CHydro
        
        self.evaluated=False
        
    def _evaluate(self):
        self.Lcoe, self.Penalties = F(self)
        self.evaluated=True


    # def __repr__(self):
    #     """S = Solution(list(np.ones(64))) >> print(S)"""
    #     return 'Solution({})'.format(self.x)

if __name__=='__main__':
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    solution = Solution(x)#/1.25) 
    
    def test():
        solution = Solution(x)#/1.25) 
        solution._evaluate()
        print(solution.Lcoe, solution.Penalties)
    test()