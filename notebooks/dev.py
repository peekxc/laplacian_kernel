import numpy as np
import cupy as cp
import cupyx 
from LaplacianOp import LaplacianFull, LaplacianSparse
from math import comb
n = 10
M,N = comb(n,3), comb(n,2)
S = np.arange(M)
F = np.arange(N)
L = LaplacianSparse(S=S, F=F, n=n, k=3, gpu=True)

import primate




# print(L)
# x = L.tm.arange(L.N, dtype=np.float32)
# # y = L.tm.asarray(L._y)
# y = cp.zeros(N)
# print(y)
# print(len(y))
# print(N)
# L.mult(x, L.deg, y)
# print(L.S)
# print(L.F)
# print(L.deg)
# print(y.device)
# print(L.launch_config)
# L.launch_config(x, y, L.S, L.F, L.n, L.k, L.BT)

# print(y)