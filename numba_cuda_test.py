import numpy as np
from numba import cuda
from numba.types import int32
import time

time.sleep(2)

@cuda.jit
def array_sum(data):
    tid = cuda.threadIdx.x
    size = len(data)
    if tid < size:
        i = cuda.grid(1)

        # Declare an array in shared memory
        shr = cuda.shared.array(nelem, int32)
        shr[tid] = data[i]

        # Ensure writes to shared memory are visible
        # to all threads before reducing
        cuda.syncthreads()

        s = 1
        while s < cuda.blockDim.x:
            if tid % (2 * s) == 0:
                # Stride by `s` and add
                shr[tid] += shr[tid + s]
            s *= 2
            cuda.syncthreads()

        # After the loop, the zeroth  element contains the sum
        if tid == 0:
            data[tid] = shr[tid]

@cuda.jit
def f(a, b, c):
    # like threadIdx.x + (blockIdx.x * blockDim.x)
    tid = cuda.grid(1)
    size = len(c)

    if tid < size:
        c[tid] = a[tid] + b[tid]

big_number = 2**10

a = cuda.to_device(np.arange(big_number))
nelem = len(a)

start = time.time()
array_sum[1, nelem](a)
print(a[0])
end = time.time()
print(f""" CUDA vectorised sum took {end-start} seconds. Result: {a[0]} """)

start = time.time()

res = np.arange(big_number).sum()
print(res)

end = time.time()
print(f""" numpy sum took {end-start} seconds. Result: {res} """)

a = cuda.to_device(np.random.random(big_number))
b = cuda.to_device(np.random.random(big_number))
c = cuda.device_array_like(a)

start = time.time()
f.forall(len(a))(a,b,c)
# print(c.copy_to_host())
end = time.time()
print(f""" CUDA vectorised add to {end-start} seconds. """ )

start = time.time()
np.random.random(big_number) + np.random.random(big_number)
end = time.time()
print(f""" numpy add too {end-start} seconds. """)

