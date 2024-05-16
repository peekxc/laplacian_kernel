import numpy as np
import cupy as cp
from LaplacianOp import LaplacianFull

L = LaplacianFull(n=100, k=3, gpu=False)
x = np.arange(L.N)
y = np.zeros(L.N)
L(x, y)

print(L)
print(f"x = {x}")
print(f"y = {y}")

print("\n-------- END CPU VERSION ----------\n")

## Cuda version 
L = LaplacianFull(n=100, k=3, gpu=True)
print(L)
print(L.deg)

x = cp.arange(L.N, dtype=np.float32)
y = cp.zeros(L.N, dtype=np.float32)
L(x, y)
cp.cuda.stream.get_current_stream().synchronize()

print(f"x = {x}")
print(f"y = {y}")

# import timeit
from cupyx.profiler import benchmark
def do_matvec():
  L(x, y)
  
print(benchmark(do_matvec, n_repeat=30))  

# L = LaplacianBenchmark(16, 4, gpu=True, n_kernels=1)
# print(L)
