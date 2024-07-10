#https://numba.discourse.group/t/using-guvectorize-inside-a-jitted-function/1966/9

import numpy as np
from numba import guvectorize, jit

@guvectorize(['(float64[:], int64, float64[:])'], '(n),()->(n)', target='cuda')
def guvec(a: np.ndarray, add: int, out: np.ndarray):
    """Generate ufunc."""
    for i in range(a.size):
        out[i] = a[i] + add

@jit('float64[:](float64[:], int64)')
def jit_guvec(a: np.ndarray, add: int) -> np.ndarray:
    """Use ufunc as inner func in jit."""
    return guvec(a, add)

arr = np.arange(5)
add = 3

print(jit_guvec(arr, add))
print(jit_guvec(arr, add))
print(jit_guvec(arr, add))