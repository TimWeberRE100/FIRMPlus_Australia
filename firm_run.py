import numpy as np
from numba import njit  # type: ignore

from firm.Costs import Raw_Costs
from firm.Input import (
    Evaluate,
    Lengths,
    Solution,
    lb,
    network_mask,
    scenario,
    ub,
    undersea_mask,
    x0,
)

if __name__ == "__main__":
    cost_model = Raw_Costs(scenario, Lengths, undersea_mask, network_mask).CostFactors()

    # x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)

    import timeit

    def do_work():
        test(x0, cost_model, True)
        x = (ub - lb) * np.random.rand(len(lb))
        test(x, cost_model)

    @njit
    def test(x, cost_model, disp=False):
        solution = Solution(x0)
        Evaluate(solution, cost_model)
        if disp:
            #          ("time_storage_behavior", float64),
            # ("time_imbalancet", float64),
            # ("time_update_soc", float64),
            print(
                "time_transmission",
                solution.time_transmission,
                "cc | time_backfill",
                solution.time_backfill,
                "cc | time_basic",
                solution.time_basic,
                "cc | time_interconnection",
                solution.time_interconnection,
                "cc | time_storage_behavior",
                solution.time_storage_behavior,
                "cc | time_imbalancet",
                solution.time_imbalancet,
                "cc | time_update_soc",
                solution.time_update_soc,
                "cc -- ",
                solution.LCOE,
                solution.Penalties,
            )

    print("Before JIT")
    test(x0, cost_model)

    n_attempts = 8
    print(f"Running timeit test now {n_attempts} attempts")

    results = timeit.timeit(lambda: do_work(), number=n_attempts)
    print(f"Timeit calculated: {results/n_attempts}")
