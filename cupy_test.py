import cupy as cp
import numpy as np
import time


@cp.fuse()
def sum_of_products(x, y):
    return cp.sum(x * y, axis=-1)


big_number = 2**12

sum_of_products(cp.arange(0, big_number), cp.arange(0, big_number))
sum_of_products(cp.arange(0, big_number), cp.arange(0, big_number))
start = time.time()
print(sum_of_products(cp.arange(0, big_number), cp.arange(0, big_number)))
end = time.time()
print(f""" CUDA took {end-start} seconds. """)

start = time.time()
print(np.sum(np.arange(big_number) * np.arange(big_number), axis=-1))
end = time.time()
print(f""" numpy took {end-start} seconds. """)

