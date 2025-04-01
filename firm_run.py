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

    @njit
    def test(x, cost_model, disp=False):
        solution = Solution(x0)
        Evaluate(solution, cost_model)
        if disp:
            print(solution.LCOE, solution.Penalties)

    test(x0, cost_model, True)
    x = (ub - lb) * np.random.rand(len(lb))
    test(x, cost_model)
