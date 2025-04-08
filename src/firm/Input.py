
import numpy as np
from numba import boolean, float64, int64, njit, types  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba.typed.typeddict import Dict as TypedDict

from firm.Costs import Raw_Costs
from firm.Simulation import Simulate
from firm.Utils import zero_safe_division

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-s', '--scenario', type=int, default=21, required=False, help='Scenario number')
args = parser.parse_args()
scenario = args.scenario    

Nodel = np.array(["FNQ", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"])
PVl = np.array(
    ["NSW"] * 7
    + ["FNQ"] * 1
    + ["QLD"] * 2
    + ["FNQ"] * 3
    + ["SA"] * 6
    + ["TAS"] * 0
    + ["VIC"] * 1
    + ["WA"] * 1
    + ["NT"] * 1
)
Windl = np.array(
    ["NSW"] * 8
    + ["FNQ"] * 1
    + ["QLD"] * 2
    + ["FNQ"] * 2
    + ["SA"] * 8
    + ["TAS"] * 4
    + ["VIC"] * 4
    + ["WA"] * 3
    + ["NT"] * 1
)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))

MLoad = np.genfromtxt("../Data/electricity.csv", delimiter=",", skip_header=1, usecols=range(4, 4 + len(Nodel))) 
MLoad /= 1000  # MW to GW

TSPV = np.genfromtxt("../Data/pv.csv", delimiter=",", skip_header=1, usecols=range(4, 4 + len(PVl)))
TSWind = np.genfromtxt("../Data/wind.csv", delimiter=",", skip_header=1, usecols=range(4, 4 + len(Windl)))

assets = np.genfromtxt("../Data/hydrobio.csv", dtype=None, delimiter=",", encoding=None)[1:, 1:].astype(float)
CHydro, CBio = (assets[:, x] * 0.001 for x in range(assets.shape[1])) # MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0])  # 24/7, GW
CPeak = CHydro + CBio - CBaseload  # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
lengths = np.array([1500, 1000, 1000, 800, 1200, 2400, 400], dtype=np.int64)
DCloss = lengths * 0.03 * 0.001  # 3% per 1000 km
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)

efficiency = 0.8

coverage = [
    np.array(["NSW", "QLD", "SA", "TAS", "VIC"]),
    np.array(["NSW", "QLD", "SA", "TAS", "VIC", "WA"]),
    np.array(["NSW", "NT", "QLD", "SA", "TAS", "VIC"]),
    np.array(["NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"]),
    np.array(["FNQ", "NSW", "QLD", "SA", "TAS", "VIC"]),
    np.array(["FNQ", "NSW", "QLD", "SA", "TAS", "VIC", "WA"]),
    np.array(["FNQ", "NSW", "NT", "QLD", "SA", "TAS", "VIC"]),
    np.array(["FNQ", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"]),
]
coverage_int = [np.array([n_node[node] for node in node_array]) for node_array in coverage]

basic_network = np.array(
    [
        [0, 3],  # FNQ-QLD
        [1, 3],  # NSW-QLD
        [1, 4],  # NSW-SA
        [1, 6],  # NSW-VIC
        [2, 4],  # NT-SA
        [4, 7],  # SA-WA
        [5, 6],  # TAS-VIC
    ],
    dtype=np.int64,
)
if scenario <= 17:
    node = Nodel_int[scenario % 10]

    MLoad = MLoad[:, Nodel_int == node]
    TSPV = TSPV[:, PVl_int == node]
    TSWind = TSWind[:, Windl_int == node]
    CHydro, CBio, CBaseload, CPeak = [x[Nodel_int == node] for x in (CHydro, CBio, CBaseload, CPeak)]

    Nodel_int, PVl_int, Windl_int = [x[x == n_node[node]] for x in (Nodel_int, PVl_int, Windl_int)]
    # Nodel, PVl, Windl = [x[x == node] for x in (Nodel, PVl, Windl)]
    basic_network=np.empty((0,0), np.int64)
    network = np.empty((0, 0, 0, 0), dtype=np.int64)
    network_mask = np.zeros(len(basic_network), dtype=np.bool_)
    directconns = np.empty((0, 0), dtype=np.int64)
    trans_mask = np.empty((0, 0), dtype=np.bool_)

elif scenario >= 21:
    coverage_int = coverage_int[scenario % 10 - 1]

    MLoad = MLoad[:, np.isin(Nodel_int, coverage_int)]
    TSPV = TSPV[:, np.isin(PVl_int, coverage_int)]
    TSWind = TSWind[:, np.isin(Windl_int, coverage_int)]
    CHydro, CBio, CBaseload, CPeak = [x[np.isin(Nodel_int, coverage_int)] for x in (CHydro, CBio, CBaseload, CPeak)]

    if 0 not in coverage_int:
        MLoad[:, np.where(coverage_int == 3)[0][0]] /= 0.9

    Nodel_int, PVl_int, Windl_int = [x[np.isin(x, coverage_int)] for x in (Nodel_int, PVl_int, Windl_int)]
    # Nodel, PVl, Windl = [x[np.isin(x, coverage)] for x in (Nodel, PVl, Windl)]

    from firm.Network import generate_network

    basic_network, network, network_mask, trans_mask, directconns, triangulars = generate_network(basic_network, Nodel_int)
    
    
    
resolution = 0.5
years = int(resolution * len(MLoad) / 8760) 
intervals = int(years * 8760 / resolution)
firstyear, finalyear, timestep = (2020, 2020 + years - 1, 1)

# MLoad, TSPV, TSWind = (x[:intervals, :] for x in (MLoad, TSPV, TSWind))

nhvi = network_mask.sum()
nodes = MLoad.shape[1]

pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx = pzones
widx = pidx + wzones
spidx = widx + nodes
seidx = spidx + nodes

energy = MLoad.sum() * 1000 * resolution / years  # MWh p.a.

lb = np.array(
    [0.0] * pzones + 
    [0.0] * wzones + 
    [0.0] * nodes + 
    [0.0] * nodes + 
    [0.0] * nhvi
    )
ub = np.array(
    [24.0] * pzones + 
    [24.0] * wzones + 
    [24.0] * nodes + 
    [600.0] * nodes + 
    [20.0] * nhvi
    )

x0 = np.concatenate(
    (
        MLoad.sum() / intervals * 0.75 / pzones / TSPV.mean(axis=0),
        MLoad.sum() / intervals * 0.75 / wzones / TSWind.mean(axis=0),
        MLoad.max(axis=0) * 1,
        MLoad.max(axis=0) * 36,
        np.repeat(MLoad.max() * 0.6, nhvi),
    )
)
x0 = np.minimum(ub, x0)

cost_model = Raw_Costs(
    scenario, 
    lengths, 
    undersea_mask, 
    network_mask
).CostFactors()

@njit 
def years_to_intervals(y:int):
    if y == -1:
        return years, intervals
    elif y <= years:
        return y, int(y*8760/resolution)
    raise Exception

    
# Specify the types for jitclass
solution_spec = [
    ("x", float64[:]),
    ("scenario", int64),
    ("intervals", int64),
    ("nodes", int64),
    ("nhvi", int64),
    ("resolution", float64),
    ("years", int64),
    ("efficiency", float64),
    ("energy", float64),
    ("Flex_res", float64),
    ("Nodel_int", int64[:]),
    # ('PVl_int', int64[:]),
    # ('Windl_int', int64[:]),
    ("networksteps", int64),
    ("network_mask", boolean[:]),
    ("network", int64[:, :, :, :]),
    ("basic_network", int64[:, :]),
    # ("directconns", int64[:, :]),
    # Capacities in GW/GWh
    ("CPV", float64[:]),
    ("CWind", float64[:]),
    ("CPHP", float64[:]),
    ("CPHS", float64[:]),
    ("CHVI", float64[:]),
    ("CBaseload", float64[:]),
    ("CPeak", float64[:]),
    ("CHydro", float64[:]),
    ("CBio", float64[:]),
    # Nodally diaggregated operations in GW/GWh
    ("MFlexible", float64[:, :]),
    ("MDischarge", float64[:, :]),
    ("MCharge", float64[:, :]),
    ("MStorage", float64[:, :]),
    ("MDeficit", float64[:, :]),
    ("MSpillage", float64[:, :]),
    ("MNetload", float64[:, :]),
    ("MImport", float64[:, :]),
    ("MPV", float64[:, :]),
    ("MWind", float64[:, :]),
    ("MLoad", float64[:, :]),
    ("MBaseload", float64[:, :]),
    ("MHydro", float64[:, :]),
    ("MBio", float64[:, :]),
    ("MUnbalanced", float64[:,:]),
    # Transmission
    ("TDC", float64[:, :]),
    ("Topology", float64[:, :]),
    ("trans_mask", boolean[:, :]),
    ("TImport", float64[:, :, :]),
    ("TExport", float64[:, :, :]),
    #Objectives
    ("Penalties", float64),
    ("LCOE", float64),
    ("LCOG", float64),
    ("LCOB", float64),
    ("LCOSP", float64),
    ("LCOSB", float64),
    ("LCOBS", float64),
    ("LCOBT", float64),
    ("LCOBL", float64),
    ("CAPEX", float64),
    ("OPEX", float64),
    
    ("cache_primary_donors", types.DictType(int64, int64[:, :])),
    ("cache_secondary_donors", types.DictType(int64, int64[:, :, :])),
    ("cache_tertiary_donors", types.DictType(int64, int64[:, :, :])),
    ("cache_quaternary_donors", types.DictType(int64, int64[:, :, :])),
    
    ("profiling", boolean),
    # time profiling
    ("time_transmission", float64),
    ("time_backfill", float64),
    ("time_basic", float64),
    ("time_interconnection0", float64),
    ("time_interconnection1", float64),
    ("time_interconnection2", float64),
    ("time_interconnection3", float64),
    ("time_storage_behavior", float64),
    ("time_storage_behaviort", float64),
    ("time_spilldef", float64),
    ("time_spilldeft", float64),
    ("time_update_soc", float64),
    ("time_update_soct", float64),
    ("time_unbalancedt", float64),
    ("time_unbalanced", float64),
    
    ("calls_transmission", int64),
    ("calls_backfill", int64),
    ("calls_basic", int64),
    ("calls_interconnection0", int64),
    ("calls_interconnection1", int64),
    ("calls_interconnection2", int64),
    ("calls_interconnection3", int64),
    ("calls_storage_behavior", int64),
    ("calls_storage_behaviort", int64),
    ("calls_spilldef", int64),
    ("calls_spilldeft", int64),
    ("calls_update_soc", int64),
    ("calls_update_soct", int64),
    ("calls_unbalancedt", int64),
    ("calls_unbalanced", int64),
]

@jitclass(solution_spec)
class Solution:
    def __init__(
            self, 
            x: np.ndarray, 
            years: int = years, 
            profiling: bool = False
            ):
        assert len(x) == len(lb)

        self.x = x

        self.scenario = scenario
        self.nodes = nodes
        self.resolution = resolution
        self.efficiency = efficiency
        self.years, self.intervals = years_to_intervals(years)
        self.energy = energy
        
        self.Nodel_int = Nodel_int
        self.network_mask = network_mask
        self.network = network
        self.basic_network = basic_network
        # self.directconns = directconns
        self.networksteps = np.where(triangulars == network.shape[2])[0][0]
        self.trans_mask = trans_mask

        self.nhvi = self.network_mask.sum()

        self.Flex_res = 20000 / self.resolution * self.years

        # self.PVl_int, self.Windl_int = PVl_int, Windl_int

        self.CPV = x[:pidx]
        self.CWind = x[pidx:widx]
        self.CPHP = x[widx:spidx]
        self.CPHS = x[spidx:seidx]
        self.CHVI = x[seidx:]
        self.CBaseload = CBaseload
        self.CPeak = CPeak
        self.CHydro = CHydro
        self.CBio = CBio

        self.cache_primary_donors = TypedDict.empty(int64, int64[:, :])
        self.cache_secondary_donors = TypedDict.empty(int64, int64[:, :, :])
        self.cache_tertiary_donors = TypedDict.empty(int64, int64[:, :, :])
        self.cache_quaternary_donors = TypedDict.empty(int64, int64[:, :, :])
        
        self.profiling = profiling
        if self.profiling:
            self.time_transmission = 0.0
            self.time_backfill = 0.0
            self.time_basic = 0.0
            self.time_interconnection0 = 0.0
            self.time_interconnection1 = 0.0
            self.time_interconnection2 = 0.0
            self.time_interconnection3 = 0.0
            self.time_storage_behavior = 0.0
            self.time_storage_behaviort = 0.0
            self.time_spilldef = 0.0
            self.time_spilldeft = 0.0
            self.time_update_soc = 0.0
            self.time_update_soct = 0.0
            self.time_unbalancedt = 0.0
            self.time_unbalanced = 0.0
            
            self.calls_transmission = 0
            self.calls_backfill = 0
            self.calls_basic = 0
            self.calls_interconnection0 = 0
            self.calls_interconnection1 = 0
            self.calls_interconnection2 = 0
            self.calls_interconnection3 = 0
            self.calls_storage_behavior = 0
            self.calls_storage_behaviort = 0
            self.calls_spilldef = 0
            self.calls_spilldeft = 0
            self.calls_update_soc = 0
            self.calls_update_soct = 0
            self.calls_unbalancedt = 0
            self.calls_unbalanced = 0
        
    def _instantiate_operations(self):
        self.MLoad = MLoad[:self.intervals, :]
        self.MPV = np.zeros((self.intervals, self.nodes))
        self.MWind = np.zeros((self.intervals, self.nodes))
        for i, n in enumerate(self.Nodel_int):
            self.MPV[:, i] += (TSPV[:self.intervals, PVl_int == n] * self.CPV[PVl_int == n]).sum(axis=1)
            self.MWind[:, i] += (TSWind[:self.intervals, Windl_int == n] * self.CWind[Windl_int == n]).sum(axis=1)

        self.MNetload = self.MLoad - self.MPV - self.MWind - self.CBaseload
        self.MUnbalanced = self.MNetload.copy()
        self.MDeficit, self.MSpillage = np.maximum(0, self.MNetload), -np.minimum(0, self.MNetload)

        self.MFlexible = np.zeros((self.intervals, self.nodes), dtype=np.float64)

        self.MDischarge = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        self.MCharge = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        self.MStorage = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        self.MStorage[-1] = 0.5 * self.CPHS

        self.TImport = np.zeros((self.intervals, self.nhvi, self.nodes), dtype=np.float64)
        self.TExport = np.zeros((self.intervals, self.nhvi, self.nodes), dtype=np.float64)
        self.TDC = np.zeros((self.intervals, self.nodes), dtype=np.float64)

#%% 

@njit
def Evaluate(S, cost_model):
    S._instantiate_operations()
    Simulate(S)

    S.Penalties = np.maximum(0, S.MDeficit.sum())*1000  # MWh/resolution

    CHVI = np.zeros(len(S.network_mask), dtype=np.float64)
    CHVI[S.network_mask] = S.CHVI

    cost = np.array(
        [
            # generation capex
            S.CPV.sum() * cost_model.pv[0],
            S.CWind.sum() * cost_model.onsw[0],
            0,  # S.CGas.sum()  * cost_model.gas[0],
            (S.CHydro.sum() + S.CBio.sum() + S.CBaseload.sum()) * cost_model.hydro[0],
            # generation fom
            S.CPV.sum() * cost_model.pv[1],
            S.CWind.sum() * cost_model.onsw[1],
            0,  # S.CGas.sum()  * cost_model.gas[1],
            (S.CHydro.sum() + S.CBio.sum() + S.CBaseload.sum()) * cost_model.hydro[1],
            # generation vom
            # pv, onsw, battery are 0
            0,  # S.GGas.sum() * S.resolution / S.years * cost_model.gas[2],
            (S.MFlexible.sum() + S.CBaseload.sum() * S.intervals) * S.resolution / S.years * cost_model.hydro[2],
            # storage
            S.CPHP.sum() * cost_model.phes[0],
            S.CPHS.sum() * cost_model.phes[1],
            S.CPHP.sum() * cost_model.phes[2],
            S.MDischarge.sum() * S.resolution / S.years * cost_model.phes[3],
            cost_model.phes[4],
        ]
        +
        # transmission network
        list(
            (
                S.CPV.sum()
                + S.CWind.sum()
                +
                # S.CGas.sum() +
                S.CHydro.sum()
                + S.CBio.sum()
            )
            * cost_model.ac
        )
        + list((CHVI * cost_model.hvi).sum(axis=1))
    )

    # Levelised Costs of:
    # Electricity
    S.LCOE = cost.sum() / S.energy
    # Generation
    S.LCOG = cost[:10].sum() / (
        1000
        * S.resolution
        / S.years
        * (
            S.MPV.sum()
            + S.MWind.sum()
            # +S.MGas.sum()
            + S.MFlexible.sum()
            + S.CBaseload.sum() * S.intervals
        )
    )
    # Storage
    # S.LCOSP = zero_safe_division(cost[10:15].sum(), S.MDischarge.sum()*S.resolution/S.years)
    # Balancing - Storage
    S.LCOBS = cost[10:15].sum() / S.energy
    # Balancing - Transmission
    S.LCOBT = cost[15:].sum() / S.energy
    # Balancing - Spillage
    S.LCOBL = S.LCOE - S.LCOG - S.LCOBS - S.LCOBT
    S.LCOB = S.LCOBS + S.LCOBT + S.LCOBL

    S.CAPEX = sum([cost[i] for i in [0, 1, 2, 3, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21]]) / S.energy
    S.OPEX = S.LCOE - S.CAPEX

    return S.LCOE, S.Penalties
