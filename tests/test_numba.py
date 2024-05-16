# %% imports
from typing import Callable
import numpy as np
import numba as nb

# %% JIT compiled functions

## Binary search over the range [bottom, top] for index 'n' satisfying choose(n, m) < r <= choose(n+1, m) 
@nb.jit(nopython=True)
def get_max(top: int, bottom: int, pred: Callable, *args):
  # if (~(BT[m][bottom] <= r)): 
  if ~pred(bottom, *args):
    return bottom
  size = (top - bottom)
  while (size > 0):
    step = size >> 1
    mid = top - step
    # if (BT[m][mid] <= r):
    if ~pred(mid, *args):
      top = mid - 1
      size -= step + 1
    else:
      size = step
  return top

@nb.jit(nopython=True)
def find_k_pred(w: int, r: int, m: int, BT: np.ndarray) -> bool:
  return BT[m][w] <= r
  
@nb.jit(nopython=True)
def get_max_vertex(r: int, m: int, n: int, BT: np.ndarray) -> int:
  k_lb: int = m - 1
  return 1 + get_max(n, k_lb, find_k_pred, r, m, BT)

@nb.jit(nopython=True)
def k_boundary(n: int, simplex: int, dim: int, BT: np.ndarray, out: np.ndarray):
  idx_below: int = simplex
  idx_above: int = 0
  j = n - 1
  for k in np.flip(np.arange(dim+1)):
    j = get_max_vertex(idx_below, k + 1, j, BT) - 1
    c = BT[k+1][j]
    face_index = idx_above - c + idx_below
    idx_below -= c
    idx_above += BT[k][j]
    out[dim-k] = face_index

@nb.jit(nopython=True)
def laplacian1_matvec(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(4, dtype=np.int32)
  for tid in range(N):
    k_boundary(n, tid, k - 1, BT, ps)
    i,j,q,_= ps
    y[i] += (x[q] - x[j])
    y[j] -= (x[j] + x[q])
    y[q] += (x[i] - x[j])

# %% Test
from combin import rank_to_comb, comb_to_rank
from itertools import combinations
from math import comb
n, k = 500, 3
BT = np.array([[comb(ni, ki) for ni in range(n)] for ki in range(k+2)]).astype(np.int64)

k_faces = np.zeros(k, dtype=np.int64)
# k_boundary(n, simplex=0, dim=2, BT=BT, out=k_faces)
k_simplices = np.array(rank_to_comb([r for r in range(comb(n, k))], k=k, n=n, order='colex'))
for r, simplex in enumerate(k_simplices):
  boundary_ranks = np.array(comb_to_rank(combinations(simplex, k-1), k=k-1, n=n, order='colex'))
  k_boundary(n, simplex=r, dim=k-1, BT=BT, out=k_faces)
  assert np.all(boundary_ranks == k_faces)

## Precmpute degree map
deg_map = { i : 0 for i in range(comb(n, k-1)) }
for r in range(comb(n, k)):
  k_boundary(n, simplex=r, dim=k-1, BT=BT, out=k_faces)
  for i in k_faces:
    deg_map[i] += 1 
deg = np.array([deg_map[r] for r in range(comb(n, k-1))])
# deg = 

BT = np.array([[comb(ni, ki) for ni in range(n)] for ki in range(k+2)]).astype(np.int64)
x = np.ones(comb(n, k-1))
y = np.zeros(comb(n, k-1))
laplacian1_matvec(x, y, n=n, k=k, N=comb(n,k), BT=BT, deg=deg)
print(y)

import timeit
timeit.timeit(lambda: laplacian1_matvec(x, y, n=n, k=k, N=comb(n,k), BT=BT, deg=deg), number = 1)

from spirit.apparent_pairs import boundary_matrix
D2 = boundary_matrix(p=2, p_simplices=np.arange(comb(n, k)), f_simplices=np.arange(comb(n, k-1)), n = n)
y_true = (D2 @ D2.T) @ x

assert np.allclose(y_true, y)

# %% For higher k, use this: 
K = 7
sgn_pattern = np.where(np.arange(K) % 2 == 0, 1, -1)
sgn_pattern = np.array([si*sj for si,sj in list(combinations(sgn_pattern, 2))])

# from simplextree import SimplexTree
# st = SimplexTree(combinations(range(n), k-1))
# st.expand(2)


# import splex as sx
# D2 = sx.boundary_matrix(st, p=2)
# (D2 @ D2.T) @ x




# %% 

[get_max_vertex(r=r, m=1, n=10, BT=BT) for r in range(comb(n, k))]

# const auto pred = [r,m](index_t w) -> bool { return BinomialCoefficient< safe >(w, m) <= r; };
# if constexpr(use_lb){
#   k_lb = find_k(r,m); // finds k such that comb(k-1, m) <= r
# } else {
#   k_lb = m-1; 
# }


# find_k_pred(0, 5, 2, BT)

get_max(100, 0, find_k_pred, 5, 2, BT)

comb()









get_max(10, 0, 123, BT)

get_max(10, 0, 123, BT)