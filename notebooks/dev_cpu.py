# %% imports
import numpy as np
from math import comb 
from comb_laplacian import LaplacianFull, LaplacianSparse
from scipy.sparse.linalg import LinearOperator

n, k = 20000, 3
L = LaplacianFull(n, k, gpu=False)
x = np.arange(L.shape[1])
print(L)
# L.tocoo().diagonal()


S = np.arange(comb(n, k))
F = np.arange(comb(n, k-1))
L = LaplacianSparse(S, F, n=n, k=k, gpu=False)
print(L)

[fm.flag_weight(offset+ii, 1, n, weights, BT) for ii in range(15)]

L @ x

# %% Form rips filtration
from comb_laplacian import compile_filtrations, compile_laplacians, flag_simplices, LaplacianSparse
from scipy.spatial.distance import pdist, cdist, squareform
from array import array 
from math import comb
# L = LaplacianFull(n, k, gpu=False)
lm = compile_laplacians(gpu=False)
fm = compile_filtrations(gpu=False)

n, k = 1000, 4
np.random.seed(1234)
X = np.random.uniform(size=(n, 2))
w = pdist(X)

# n=1000,k=4 <=> 10m + 42s 
S = flag_simplices(w, p=3, eps=3.0, discard_ap=True, n_blocks=1024, verbose=True)
F = flag_simplices(w, p=2, eps=3.0, discard_ap=False, n_blocks=1024, verbose=True)
F = np.sort(F)
L = LaplacianSparse(S,F,n,k)
print(L)
# Up-Laplacian (166167000 / 176064040) - (2/3)-simplices 
# n: 1000, k: 4, Memory usage: 4.4GB 

import primate
from primate.trace import hutch
from bokeh.io import output_notebook
output_notebook()

hutch(L, fun='smoothstep', a=0.0, b=1e-6, max_iter = 200, info=True, verbose=True, deg=20, orth=8, ncv=10)

## Note that b matters a lot! Making b too small can greatly affect the error rates!
A = L.tocoo().todense() / 50.0
hutch(A, fun='smoothstep', a=1e-6, b=.40, max_iter = 2400, info=False, verbose=True, deg=20, orth=8, ncv=32, plot=True)

hutch(A, fun='smoothstep', a=1e-6, b=1e-2, max_iter = 2400, info=False, verbose=True, deg=20, orth=8, ncv=32, plot=True)




from scipy.sparse import eigvalsh
from primate.diagonalize import lanczos
# lanczos(A, deg=40)

import timeit
L_sp = L.tocoo()
x = np.random.uniform(size=L.shape[1])
timeit.timeit(lambda: L @ x, number=150)
timeit.timeit(lambda: A @ x, number=150)
timeit.timeit(lambda: L_sp @ x, number=150)


hutch(A, fun='smoothstep', a=1e-3, b=0.0014327613*10.0, max_iter = 2400, info=False, verbose=True, deg=80, orth=30, ncv=32, plot=True)


ew = np.linalg.eigvalsh(A)
len(ew[ew > 1e-6])


np.max(ew[ew > 1e-6])
np.min(ew[ew > 1e-6])

np.histogram(np.linalg.eigvalsh(A))

np.linalg.matrix_rank(A)

# BT = np.array([[int(comb(ni, ki)) for ni in range(n+1)] for ki in range(k+2)]).astype(np.int64)
# assert np.max(S) < comb(n,k)
# assert np.max(F) < comb(n,k-1)
# F = np.sort(F)
# k_faces = np.zeros(k, dtype=np.int64)
# lm.k_boundary_cpu(simplex=19204, dim=k-1, n=n, BT=BT, out=k_faces)
# comb(n,k-1)

# from combin import rank_to_comb, comb_to_rank
# rank_to_comb(19204, k=k, n=n, order='colex')
# w[comb_to_rank([0,40], k=k, n=n, order='lex')]
# w[comb_to_rank([40,49], k=k, n=n, order='lex')]
# w[comb_to_rank([0,49], k=k, n=n, order='lex')]

# comb_to_rank([40,49], k=k, n=n, order='colex')

# lm.sp_precompute_deg(n, k, S, F, BT)

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



