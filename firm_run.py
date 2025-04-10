import numpy as np
from numba import njit, prange  # type: ignore
from psutil import cpu_count

from firm.Input import (
    Evaluate,
    Solution,
    cost_model,
    lb,
    lengths,
    network_mask,
    scenario,
    ub,
    undersea_mask,
    x0,
    zero_safe_division,
)

from firm.Benchmark import (
    Benchmark, 
    test, 
    profile, 
)

if __name__ == "__main__":    

    import timeit

    # print("Before JIT")
    profile(x0, cost_model)
    test(x0, cost_model)

    n_attempts = 3
    n_parallel = 3
    n_eval = n_parallel*cpu_count(logical=True)
    print(f"Running timeit test now {n_attempts} attempts of {n_eval} in parallel")

    results = timeit.timeit(lambda: Benchmark(n_eval), number=n_attempts)/n_attempts
    print(f"Timeit calculated: {results} per batch of n_eval")
    print(f"\t{results/n_parallel} per parallel batch")
    print(f"\t{results/n_eval} per single eval")
