from typing import Callable
from numba import int64

import numpy as np
import numba as nb
do_bounds_check = True

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def get_max(top: int, bottom: int, pred: Callable, *args):
  if ~pred(bottom, *args):
    return bottom
  size = (top - bottom)
  while (size > 0):
    step = size >> 1
    mid = top - step
    if ~pred(mid, *args):
      top = mid - 1
      size -= step + 1
    else:
      size = step
  return top

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def find_k_pred(w: int, r: int, m: int, BT: np.ndarray) -> bool:
  return BT[m][w] <= r

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def get_max_vertex(r: int, m: int, n: int, BT: np.ndarray) -> int:
  k_lb: int = m - 1
  return 1 + get_max(n, k_lb, find_k_pred, r, m, BT)

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def comb_to_rank_lex2(i: int, j: int, n: int) -> int:
  i, j = (j, i) if j < i else (i, j)
  return int64(n*i - i*(i+1)/2 + j - i - 1)

@nb.jit(device=True)
def rank_to_comb_colex(simplex: int, n: int, k: int, BT: np.ndarray, out: np.ndarray):
  K: int64 = int64(n - 1)
  for ki in range(1, k):
    m = int64(k - ki + 1)
    K = get_max_vertex(simplex, m, n, BT)
    # assert comb(K-1,m) <= simplex and simplex < comb(K, m)
    out[ki-1] = K-1
    simplex -= BT[m][K-1]
  out[-1] = simplex

@nb.jit(nopython=True,boundscheck=do_bounds_check)
def k_boundary_cpu(n: int, simplex: int, dim: int, BT: np.ndarray, out: np.ndarray):
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