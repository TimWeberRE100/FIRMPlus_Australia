from numba import njit
import numpy as np

@njit
def zero_safe_division(numerator, denominator, error=0):
    return error if denominator == 0 else numerator / denominator

@njit 
def array_max(arr):
    max = -np.inf
    for v in arr.ravel():
        if v > max:
            max = v 
    return max

@njit 
def array_min(arr):
    min = np.inf
    for v in arr.ravel():
        if v < min:
            min = v 
    return min


import ctypes
from ctypes.util import find_library
import platform

from numba import njit  # type: ignore


if platform.system() == "Windows":
    from ctypes.util import find_msvcrt
    __LIB = find_msvcrt()
    if __LIB is None:
        __LIB = "msvcrt.dll"
    clock = ctypes.CDLL(__LIB).clock
    clock.argtypes = []
    @njit
    def cclock():
        return clock()/1000 #cpu-seconds
 
else:
    __LIB = find_library("c")
    clock = ctypes.CDLL(__LIB).clock
    clock.argtypes = []
    
    @njit
    def cclock():
        return clock()  # cpu-cycles
