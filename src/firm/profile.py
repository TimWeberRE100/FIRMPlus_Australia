import ctypes
from ctypes.util import find_library

from numba import njit  # type: ignore

__LIB = find_library("c")
clock = ctypes.CDLL(__LIB).clock
clock.argtypes = []


@njit
def cclock():
    return clock() / 10_000  # cpu-seconds
