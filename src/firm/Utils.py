from numba import njit
import numpy as np

@njit
def zero_safe_division(numerator, denominator, error=0):
    return error if denominator == 0 else numerator / denominator

@njit
def itemwise_minimum_2d_0d(arr1, scal, retarr):
    for i in range(retarr.shape[0]):
        for j in range(retarr.shape[1]):
            retarr[i, j] = min(arr1[i, j], scal)
    return retarr

@njit
def itemwise_maximum_2d_0d(arr1, scal, retarr):
    for i in range(retarr.shape[0]):
        for j in range(retarr.shape[1]):
            retarr[i, j] = max(arr1[i, j], scal)
    return retarr

@njit
def itemwise_minimum_2d_1d(arr1, arr2, retarr):
    for i in range(retarr.shape[0]):
        for j in range(retarr.shape[1]):
            retarr[i, j] = min(arr1[i, j], arr2[j])
    return retarr

@njit
def itemwise_maximum_2d_1d(arr1, arr2, retarr):
    for i in range(retarr.shape[0]):
        for j in range(retarr.shape[1]):
            retarr[i, j] = max(arr1[i, j], arr2[j])
    return retarr

@njit
def itemwise_minimum_1d_1d(arr1, arr2, retarr):
    for i in range(len(retarr)):
        retarr[i] = min(arr1[i], arr2[i])
    return retarr

@njit
def itemwise_maximum_1d_1d(arr1, arr2, retarr):
    for i in range(len(retarr)):
        retarr[i] = max(arr1[i], arr2[i])
    return retarr

@njit
def itemwise_minimum_1d_0d(arr1, scal, retarr):
    for i in range(len(retarr)):
        retarr[i] = min(arr1[i], scal)
    return retarr

@njit
def itemwise_maximum_1d_0d(arr1, scal, retarr):
    for i in range(len(retarr)):
        retarr[i] = max(arr1[i], scal)
    return retarr

@njit
def array_sum_2d_axis0(arr):
    """like np.ndarray.sum(axis=0)"""
    ret = np.zeros(arr.shape[1], arr.dtype)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ret[j] += arr[i, j]
    return ret

@njit
def array_sum_2d_axis1(arr):
    """like np.ndarray.sum(axis=1)"""
    ret = np.zeros(arr.shape[0], arr.dtype)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ret[i] += arr[i, j]
    return ret

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

@njit 
def array_max_2d_axis0(arr):
    """ like arr.max(axis=0) for 2d arr"""
    max = arr[0].copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > max[j]:
                max[j] = arr[i, j]
    return max

@njit 
def array_max_2d_axis1(arr):
    """ like arr.max(axis=1) for 2d arr"""
    max = arr[:, 0].copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > max[i]:
                max[i] = arr[i, j]
    return max




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
        return clock() #cpu-cycles
 
else:
    __LIB = find_library("c")
    clock = ctypes.CDLL(__LIB).clock
    clock.argtypes = []
    
    @njit
    def cclock():
        return clock()  # cpu-cycles
