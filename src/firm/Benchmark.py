import numpy as np
from numba import njit, prange  # type: ignore
from time import perf_counter

from firm.Utils import zero_safe_division
from firm.Input import (
    Evaluate,
    Solution,
    Solution_data,
)


def Benchmark(n, sd, cost_model):
    _benchmark(n, sd, cost_model)

@njit(parallel=True)
def _benchmark(n, sd, cost_model):
    result = np.empty(n)
    for j in prange(n):
        result[j] = test(sd.x0, sd, cost_model)

@njit
def test(x, sd, cost_model):
    solution = Solution(x, sd)
    Evaluate(solution, cost_model)
    return solution.LCOE+solution.Penalties

def profile(x, 
            solution_data,
            cost_model, 
            disp: bool = True, 
            ):
    solution = Solution(x, solution_data)
    start = perf_counter()
    Evaluate(solution, cost_model)
    time = perf_counter() - start

    profiles = [item[5:] for item in dir(solution) if item.startswith('time_')]
    for item in profiles:
        setattr(
            solution, 
            'time_'+item, 
            getattr(solution, 'time_'+item) - solution.profile_overhead*getattr(solution, 'calls_'+item)
            )

    cputime = sum((getattr(solution, item) for item in dir(solution) if item.startswith('time_')))
    profiletime = sum((getattr(solution, item) for item in dir(solution) if item.startswith('calls_')))
    profiletime *= solution.profile_overhead
    ctwt = zero_safe_division(time,(cputime+profiletime))

    if disp:
        #          ("time_storage_behavior", float64),
        # ("time_imbalancet", float64),
        # ("time_update_soc", float64),
        print(
            "transmission\t\t",
            solution.calls_transmission, "\t|", 
            ctwt*solution.time_transmission, "\t|", 
            ctwt*zero_safe_division(
                solution.time_transmission,
                solution.calls_transmission),
            "cc \n backfill\t\t\t",
            solution.calls_backfill, "\t|", 
            ctwt*solution.time_backfill, "\t|", 
            ctwt*zero_safe_division(
                solution.time_backfill, 
                solution.calls_backfill),
            # "cc \n basic\t\t\t\t",
            # solution.calls_basic, "\t|", 
            # ctwt*solution.time_basic, "\t|", 
            # ctwt*zero_safe_division(
            #     solution.time_basic, 
            #     solution.calls_basic),
            "cc \n interconnection\t\t",
            (
                solution.calls_interconnection0
                +solution.calls_interconnection1
                +solution.calls_interconnection2
                +solution.calls_interconnection3), "\t|", 
            ctwt*(
                solution.time_interconnection0
                +solution.time_interconnection1
                +solution.time_interconnection2
                +solution.time_interconnection3), "\t|", 
            ctwt*zero_safe_division(
                solution.time_interconnection0
                +solution.time_interconnection1
                +solution.time_interconnection2
                +solution.time_interconnection3, 
                solution.calls_interconnection0
                +solution.calls_interconnection1
                +solution.calls_interconnection2
                +solution.calls_interconnection3),
            "cc \n interconnection0\t\t",
            solution.calls_interconnection0, "\t|", 
            ctwt*solution.time_interconnection0, "\t|", 
            ctwt*zero_safe_division(
                solution.time_interconnection0, 
                solution.calls_interconnection0),
            "cc \n interconnection1\t\t",
            solution.calls_interconnection1, "\t|", 
            ctwt*solution.time_interconnection1, "\t|", 
            ctwt*zero_safe_division(
                solution.time_interconnection1, 
                solution.calls_interconnection1),
            "cc \n interconnection2\t\t",
            solution.calls_interconnection2, "\t|", 
            ctwt*solution.time_interconnection2, "\t|", 
            ctwt*zero_safe_division(
                solution.time_interconnection2, 
                solution.calls_interconnection2),
            "cc \n interconnection3\t\t",
            solution.calls_interconnection3, "\t|", 
            ctwt*solution.time_interconnection3, "\t|", 
            ctwt*zero_safe_division(
                solution.time_interconnection3, 
                solution.calls_interconnection3),
            "cc \n storage_behavior\t",
            solution.calls_storage_behavior, "\t|", 
            ctwt*solution.time_storage_behavior, "\t|", 
            ctwt*zero_safe_division(
                solution.time_storage_behavior, 
                solution.calls_storage_behavior),
            "cc \n storage_behaviort\t",
            solution.calls_storage_behaviort, "\t|", 
            ctwt*solution.time_storage_behaviort, "\t|", 
            ctwt*zero_safe_division(
                solution.time_storage_behaviort, 
                solution.calls_storage_behaviort),
            "cc \n spilldef\t\t\t",
            solution.calls_spilldef, "\t|", 
            ctwt*solution.time_spilldef, "\t|", 
            ctwt*zero_safe_division(
                solution.time_spilldef, 
                solution.calls_spilldef),
            "cc \n spilldeft\t\t\t",
            solution.calls_spilldeft, "\t|",
            ctwt*solution.time_spilldeft, "\t|", 
            ctwt*zero_safe_division(
                solution.time_spilldeft, 
                solution.calls_spilldeft),
            "cc \n update_soc\t\t\t",
            solution.calls_update_soc, "\t|", 
            ctwt*solution.time_update_soc, "\t|", 
            ctwt*zero_safe_division(
                solution.time_update_soc, 
                solution.calls_update_soc),
            "cc \n update_soct\t\t\t",
            solution.calls_update_soct, "\t|", 
            ctwt*solution.time_update_soct, "\t|", 
            ctwt*zero_safe_division(
                solution.time_update_soct, 
                solution.calls_update_soct),
            "cc \n unbalancedt\t\t\t",
            solution.calls_unbalancedt, "\t|", 
            ctwt*solution.time_unbalancedt, "\t|", 
            ctwt*zero_safe_division(
                solution.time_unbalancedt, 
                solution.calls_unbalancedt),
            "cc \n unbalanced\t\t\t",
            solution.calls_unbalanced, "\t|", 
            ctwt*solution.time_unbalanced, "\t|", 
            ctwt*zero_safe_division(
                solution.time_unbalanced, 
                solution.calls_unbalanced),
            "cc \n -- ",
            solution.LCOE,
            solution.Penalties,
        )
    return solution, time, ctwt
