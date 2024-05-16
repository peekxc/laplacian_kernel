from typing import Callable
from math import comb
from numba import float32, int64
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