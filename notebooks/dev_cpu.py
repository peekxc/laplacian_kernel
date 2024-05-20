# %% imports
import numpy as np
from math import comb 
from comb_laplacian import LaplacianFull, LaplacianSparse
from scipy.sparse.linalg import LinearOperator

n, k = 50, 3
L = LaplacianFull(n, k, gpu=False)
print(L)
L.tocoo().diagonal()

x = np.arange(L.shape[1])
L @ x

L @ np.eye(L.shape[1])


S = np.arange(comb(n, k))
F = np.arange(comb(n, k-1))
L = LaplacianSparse(S, F, n=n, k=k, gpu=False)
print(L)

L @ x

# %% Form rips filtration
from comb_laplacian import compile_filtrations, flag_simplices
from scipy.spatial.distance import pdist, cdist, squareform
from array import array 
# L = LaplacianFull(n, k, gpu=False)
fm = compile_filtrations(gpu=False)

n, k = 1500, 3
np.random.seed(1234)
X = np.random.uniform(size=(n, 2))
w = pdist(X)

s = flag_simplices(w, p=2, eps=3.0, discard_ap=True, verbose=True)

## Single approach
# S_out = -1 * np.ones(comb(n,k), dtype=np.int64)
# fm.construct_flag_dense_ap(comb(n,k), 2, L.n, eps=3.0, weights=w, BT=L._BT, S=S_out, offset=0)
# print(len(S_out[S_out != -1]))

# # Blocked approach 
# S_ap = array('q')
# b = 128
# ns_per_block = comb(n,k) // b
# S_out = -1 * np.ones(ns_per_block, dtype=np.int64)
# for bi in reversed(range(b)):
#   offset = bi*ns_per_block
#   n_launches = min(comb(n,k), offset + ns_per_block) - offset
#   S_out.fill(-1)
#   fm.construct_flag_dense_ap(n_launches, 2, L.n, eps=2*np.median(w), weights=w, BT=L._BT, S=S_out, offset=offset)
#   new_s = S_out[S_out != -1]
#   print(f"bi: {bi}, # sim: {len(new_s)}")
#   S_ap.extend(new_s)
#   if len(new_s) == 0: 
#     break ## Shortcut that is unclear why it works

# print(len(S_ap))

# F_out = -1 * np.ones(comb(n,k-1), dtype=np.int64)
# fm.construct_flag_dense(n_launches, 1, L.n, eps=2*np.median(w), weights=w, BT=L._BT, S=F_out)
# F = F_out[F_out != -1]


L = LaplacianSparse(S_ap, F, n=n, k=k, gpu=False)
x = np.random.uniform(size=L.shape[1])
np.sum(L @ x == 0)




# S_eps = S_out[~np.isnan(S_out)]
# S_out = -1 * np.ones(len(L.S), dtype=np.int64)
# construct_flag_dense_ap(L.M, 1, L.n, eps=0.50, weights=w.astype(np.float32), BT=L._BT, S=S_out)
# len(S_out[S_out != -1])

# N: int, dim: int, n: int, eps: float, weights: np.ndarray, BT: np.ndarray, S: np.ndarray

# import timeit
# timeit.timeit(lambda: L @ x, number=100) # (100,3) x 100 => 7.65

# %% Rips Laplacian 
from comb_laplacian import rips_laplacian
n, k = 50, 3
np.random.seed(1234)
X = np.random.uniform(size=(n, 2))
w = pdist(X)
rips_laplacian
