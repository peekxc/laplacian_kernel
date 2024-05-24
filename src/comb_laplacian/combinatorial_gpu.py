from typing import Callable
from math import comb
from numba import cuda, int32, int64, void, float32, float64
import numpy as np
import numba as nb
import cupy as cp

# @cuda.jit(int64(int64, int64, int64, int64, int64[:,:]), device=True)
@cuda.jit(device=True)
def get_max_cuda(top: int, bottom: int, r: int, m: int, BT: np.ndarray) -> int:
  # if ~pred(bottom, *args):
  if not(BT[m][bottom] <= r):
    return bottom
  size = (top - bottom)
  while (size > 0):
    step = size >> 1
    mid = top - step
    if not(BT[m][mid] <= r):
      top = mid - 1
      size -= step + 1
    else:
      size = step
  return top

# @cuda.jit(int64(int64, int64, int64, int64[:,:]), device=True)
@cuda.jit(device=True)
def get_max_vertex_cuda(r: int, m: int, n: int, BT: np.ndarray) -> int64:
  k_lb: int64 = int64(m - 1)
  # return 1 + get_max(n, k_lb, find_k_pred, r, m, BT)
  return int64(1 + get_max_cuda(n, k_lb, r, m, BT))

# @cuda.jit(int64(int64, int64, int64), device=True)
@cuda.jit(device=True)
def comb_to_rank_lex2(i: int, j: int, n: int) -> int:
  i, j = (j, i) if j < i else (i, j)
  return int64(n*i - i*(i+1)/2 + j - i - 1)

# @cuda.jit(void(int64, int64, int64, int64[:,:], int64[:]), device=True)
@cuda.jit(device=True)
def rank_to_comb_colex(simplex: int, n: int, k: int, BT: np.ndarray, out: np.ndarray):
  K: int64 = int64(n - 1)
  for ki in range(1, k):
    m = int64(k - ki + 1)
    K = get_max_vertex_cuda(simplex, m, n, BT)
    # assert comb(K-1,m) <= simplex and simplex < comb(K, m)
    out[ki-1] = K-1
    simplex -= BT[m][K-1]
  out[k-1] = simplex

# @cuda.jit(void(int64, int64, int64, int64[:,:], int64[:]), device=True)
@cuda.jit(device=True)
def k_boundary_cuda(simplex: int, dim: int, n: int, BT: np.ndarray, out: np.ndarray):
  idx_below: int = simplex
  idx_above: int = 0
  j = n - 1
  for kr in range(dim+1):
    k = dim - kr
    j = get_max_vertex_cuda(idx_below, k + 1, j, BT) - 1
    c = BT[k+1][j]
    face_index = idx_above - c + idx_below
    idx_below -= c
    idx_above += BT[k][j]
    out[kr] = face_index
    
# @cuda.jit(void(int64, int64, int64, int64[:,:], int64[:]), device=True)
@cuda.jit(device=True)
def k_coboundary_cuda(simplex: int, dim: int, n: int, BT: np.ndarray, out: np.ndarray):
  idx_below: int64 = int64(simplex)
  idx_above: int64 = int64(0)
  j: int64 = int64(n - 1)
  k: int64 = int64(dim + 1)
  c: int64 = int64(0)
  while j >= k:
    while BT[k][j] <= idx_below:
      idx_below -= BT[k][j]
      idx_above += BT[k+1][j]
      j -= 1
      k -= 1
      # assert k != -1, "Coboundary enumeration failed"
    cofacet_index = idx_above + BT[k+1][j] + idx_below
    # print(f"{cofacet_index} = {idx_above} + {BT[k+1][j]} + {idx_below} (j={j},k={k})")
    j -= 1
    out[c] = cofacet_index
    c += 1