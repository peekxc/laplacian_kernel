# %% imports
import numpy as np
from math import comb 
from comb_laplacian import LaplacianFull, LaplacianSparse
from scipy.sparse.linalg import LinearOperator

n, k = 5, 2
L = LaplacianFull(n, k, gpu=False)
print(L)
x = np.arange(L.shape[1])
L @ x

L @ np.eye(L.shape[1])


S = np.arange(comb(n, k))
F = np.arange(comb(n, k-1))
L = LaplacianSparse(S, F, n=n, k=k, gpu=False)
print(L)

L @ np.eye(L.shape[1])

L @ x

import timeit
timeit.timeit(lambda: L @ x, number=100) # (100,3) x 100 => 7.65
# w/ parallel => 1.13 

