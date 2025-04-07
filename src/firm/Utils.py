from numba import njit

@njit
def zero_safe_division(numerator, denominator, error=0):
    return error if denominator == 0 else numerator / denominator


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
        return clock() / 10_000  # cpu-seconds?
