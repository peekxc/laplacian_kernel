#!/usr/bin/python
import numpy as np
import numba as nb
import cupy as cp

print("JIT-compiling CPU kernels...")
from laplacian_cpu import *

print("JIT-compiling GPU kernels...")
from laplacian_gpu import *


print("--- Creating benchmark ---")
xs
## Do a single matvec
# x = L.tm.arange(L.N)
# y = L.tm.zeros(L.N)
# L(x,y)

import timeit

timings = {}
for n in [16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1024, 1448]:
  for k in [3,4,5,6,7]:
    if predict_GB(n,k) > 30:
      continue
    L = LaplacianFull(n, k, gpu=True)
    x = L.tm.arange(L.N)
    y = L.tm.zeros(L.N)
    print((n,k))
    timings[(n,k)] = timeit.repeat(lambda: L(x,y), number=1, repeat=30)
  print(f"Finished: n={n}")

from json import dumps, dump
timings_ = { str(k) : v for k,v in timings.items()}
with open('timings.json', 'w') as fp:
    dump(timings_, fp)