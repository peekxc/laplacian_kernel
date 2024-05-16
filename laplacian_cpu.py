from typing import Callable
from math import comb
from numba import float32, int64
import numpy as np
import numba as nb
do_bounds_check = True

from combinatorial_cpu import *

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def laplacian0_matvec(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(2, dtype=np.int32)
  for tid in range(N):
    k_boundary_cpu(n, tid, k - 1, BT, ps)
    a,b = ps
    y[a] -= x[b]
    y[b] -= x[a]

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def laplacian1_matvec(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(4, dtype=np.int32)
  for tid in range(N):
    k_boundary_cpu(n, tid, k - 1, BT, ps)
    i,j,q,_= ps
    y[i] += (x[q] - x[j])
    y[j] -= (x[j] + x[q])
    y[q] += (x[i] - x[j])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def laplacian2_matvec(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(4, dtype=np.int32)
  for tid in range(N):
    k_boundary_cpu(n, tid, k - 1, BT, ps)
    a,b,c,d = ps
    y[a] += x[c] - (x[b] + x[d])
    y[b] += x[d] - (x[a] + x[c])
    y[c] += x[a] - (x[b] + x[d])
    y[d] += x[b] - (x[a] + x[c])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def laplacian3_matvec(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(5, dtype=np.int32)
  for tid in range(N):
    k_boundary_cpu(n, tid, k - 1, BT, ps)
    a,b,c,d,e = ps
    y[a] += (x[c] + x[e]) - (x[b] + x[d])
    y[b] += x[d] - (x[a] + x[c] + x[e])
    y[c] += (x[a] + x[e]) - (x[b] + x[d])
    y[d] += x[b] - (x[a] + x[c] + x[e])
    y[e] += (x[a] + x[c]) - (x[b] + x[d])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def laplacian4_matvec(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(6, dtype=np.int32)
  for tid in range(N):
    k_boundary_cpu(n, tid, k - 1, BT, ps)
    a,b,c,d,e,f = ps
    y[a] += x[c] + x[e] - (x[b] + x[d] + x[f])
    y[b] += x[d] + x[f] - (x[a] + x[c] + x[e])
    y[c] += x[a] + x[e] - (x[b] + x[d] + x[f])
    y[d] += x[b] + x[f] - (x[a] + x[c] + x[e])
    y[e] += x[a] + x[c] - (x[b] + x[d] + x[f])
    y[f] += x[b] + x[d] - (x[a] + x[c] + x[e])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def laplacian5_matvec(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(7, dtype=np.int32)
  for tid in range(N):
    k_boundary_cpu(n, tid, k - 1, BT, ps)
    a,b,c,d,e,f,g = ps
    y[a] += x[c] + x[e] + x[g] - (x[b] + x[d] + x[f])
    y[b] += x[d] + x[f] - (x[a] + x[c] + x[e] + x[g])
    y[c] += x[a] + x[e] + x[g] - (x[b] + x[d] + x[f])
    y[d] += x[b] + x[f] - (x[a] + x[c] + x[e] + x[g])
    y[e] += x[a] + x[c] + x[g] - (x[b] + x[d] + x[f])
    y[f] += x[b] + x[d] - (x[a] + x[c] + x[e] + x[g])
    y[g] += x[a] + x[c] + x[e] - (x[b] + x[d] + x[f])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def precompute_deg(n: int, k: int, N: int, M: int, BT: np.ndarray) -> np.ndarray:
  deg = np.zeros(N)
  k_faces = np.zeros(k, dtype=np.int32)
  for r in range(M):
    k_boundary_cpu(n, simplex=r, dim=k-1, BT=BT, out=k_faces)
    deg[k_faces] += 1
  return deg

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def sp_laplacian0_matvec(x: np.ndarray, y: np.ndarray, S: np.ndarray, F: np.ndarray, n: int, k: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(2, dtype=np.int32)
  for s in S:
    k_boundary_cpu(n, s, k - 1, BT, ps)
    a,b = np.searchsorted(F, ps)
    y[a] -= x[b]
    y[b] -= x[a]

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def sp_laplacian1_matvec(x: np.ndarray, y: np.ndarray, S: np.ndarray, F: np.ndarray, n: int, k: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(3, dtype=np.int32)
  for s in S:
    k_boundary_cpu(n, s, k - 1, BT, ps)
    a,b,c = np.searchsorted(F, ps)
    y[a] += (x[c] - x[b])
    y[b] -= (x[b] + x[c])
    y[c] += (x[a] - x[b])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def sp_laplacian2_matvec(x: np.ndarray, y: np.ndarray, S: np.ndarray, F: np.ndarray, n: int, k: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(4, dtype=np.int32)
  for s in S:
    k_boundary_cpu(n, s, k - 1, BT, ps)
    a,b,c,d = np.searchsorted(F, ps)
    y[a] += x[c] - (x[b] + x[d])
    y[b] += x[d] - (x[a] + x[c])
    y[c] += x[a] - (x[b] + x[d])
    y[d] += x[b] - (x[a] + x[c])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def sp_laplacian3_matvec(x: np.ndarray, y: np.ndarray, S: np.ndarray, F: np.ndarray, n: int, k: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(5, dtype=np.int32)
  for s in S:
    k_boundary_cpu(n, s, k - 1, BT, ps)
    a,b,c,d,e = np.searchsorted(F, ps)
    y[a] += (x[c] + x[e]) - (x[b] + x[d])
    y[b] += x[d] - (x[a] + x[c] + x[e])
    y[c] += (x[a] + x[e]) - (x[b] + x[d])
    y[d] += x[b] - (x[a] + x[c] + x[e])
    y[e] += (x[a] + x[c]) - (x[b] + x[d])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def sp_laplacian4_matvec(x: np.ndarray, y: np.ndarray, S: np.ndarray, F: np.ndarray, n: int, k: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(6, dtype=np.int32)
  for s in S:
    k_boundary_cpu(n, s, k - 1, BT, ps)
    a,b,c,d,e,f = np.searchsorted(F, ps)
    y[a] += x[c] + x[e] - (x[b] + x[d] + x[f])
    y[b] += x[d] + x[f] - (x[a] + x[c] + x[e])
    y[c] += x[a] + x[e] - (x[b] + x[d] + x[f])
    y[d] += x[b] + x[f] - (x[a] + x[c] + x[e])
    y[e] += x[a] + x[c] - (x[b] + x[d] + x[f])
    y[f] += x[b] + x[d] - (x[a] + x[c] + x[e])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def sp_laplacian5_matvec(x: np.ndarray, y: np.ndarray, S: np.ndarray, F: np.ndarray, n: int, k: int, BT: np.ndarray, deg: np.ndarray):
  np.multiply(x, deg, y) # y = x * deg
  ps = np.zeros(7, dtype=np.int32)
  for s in S:
    k_boundary_cpu(n, s, k - 1, BT, ps)
    a,b,c,d,e,f,g = np.searchsorted(F, ps)
    y[a] += x[c] + x[e] + x[g] - (x[b] + x[d] + x[f])
    y[b] += x[d] + x[f] - (x[a] + x[c] + x[e] + x[g])
    y[c] += x[a] + x[e] + x[g] - (x[b] + x[d] + x[f])
    y[d] += x[b] + x[f] - (x[a] + x[c] + x[e] + x[g])
    y[e] += x[a] + x[c] + x[g] - (x[b] + x[d] + x[f])
    y[f] += x[b] + x[d] - (x[a] + x[c] + x[e] + x[g])
    y[g] += x[a] + x[c] + x[e] - (x[b] + x[d] + x[f])

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def sp_precompute_deg(n: int, k: int, S: np.ndarray, F: np.ndarray, BT: np.ndarray) -> np.ndarray:
  deg = np.zeros(len(F))
  k_faces = np.zeros(k, dtype=np.int32)
  for s in S:
    k_boundary_cpu(n, simplex=s, dim=k-1, BT=BT, out=k_faces)
    deg[np.searchsorted(F, k_faces)] += 1
  return deg