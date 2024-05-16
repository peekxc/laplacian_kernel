from typing import Callable
from math import comb
from numba import float32, int64, cuda
import numpy as np
import numba as nb
import cupy as cp

from combinatorial_gpu import k_boundary_cuda

@cuda.jit
def laplacian1_matvec_cuda(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  ps = cuda.local.array(shape=(4,), dtype=int64)

  if tid < N:
    k_boundary_cuda(tid, k - 1, n, BT, ps)
    i, j, q = ps[0], ps[1], ps[2]
    cuda.atomic.add(y, i, x[q] - x[j])
    cuda.atomic.add(y, j, -(x[j] + x[q]))
    cuda.atomic.add(y, q, x[i] - x[j])

@cuda.jit
def laplacian2_matvec_cuda(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  tid = (cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x)
  ps = cuda.local.array(shape=(4,), dtype=int64)

  if tid < N:
    k_boundary_cuda(tid, k - 1, n, BT, ps)
    a,b,c,d = ps[0], ps[1], ps[2], ps[3]
    cuda.atomic.add(y, a, x[c] - (x[b] + x[d]))
    cuda.atomic.add(y, b, x[d] - (x[a] + x[c]))
    cuda.atomic.add(y, c, x[a] - (x[b] + x[d]))
    cuda.atomic.add(y, d, x[b] - (x[a] + x[c]))


@cuda.jit
def laplacian3_matvec_cuda(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  # cp.multiply(x, deg, y) # y = x * deg
  tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  ps = cuda.local.array(shape=(5,), dtype=int64)
  if tid < N:
    k_boundary_cuda(tid, k - 1, n, BT, ps)
    a,b,c,d,e = ps[0], ps[1], ps[2], ps[3], ps[4]
    cuda.atomic.add(y, a, (x[c] + x[e]) - (x[b] + x[d]))
    cuda.atomic.add(y, b, x[d] - (x[a] + x[c] + x[e]))
    cuda.atomic.add(y, c, (x[a] + x[e]) - (x[b] + x[d]))
    cuda.atomic.add(y, d, x[b] - (x[a] + x[c] + x[e]))
    cuda.atomic.add(y, e, (x[a] + x[c]) - (x[b] + x[d]))

@cuda.jit
def laplacian4_matvec_cuda(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  # tid = cuda.grid(1)
  tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  ps = cuda.local.array(shape=(6,), dtype=int64)
  if tid < N:
    k_boundary_cuda(tid, k - 1, n, BT, ps)
    a,b,c,d,e,f = ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]
    cuda.atomic.add(y, a, x[c] + x[e] - (x[b] + x[d] + x[f]))
    cuda.atomic.add(y, b, x[d] + x[f] - (x[a] + x[c] + x[e]))
    cuda.atomic.add(y, c, x[a] + x[e] - (x[b] + x[d] + x[f]))
    cuda.atomic.add(y, d, x[b] + x[f] - (x[a] + x[c] + x[e]))
    cuda.atomic.add(y, e, x[a] + x[c] - (x[b] + x[d] + x[f]))
    cuda.atomic.add(y, f, x[b] + x[d] - (x[a] + x[c] + x[e]))

@cuda.jit
def laplacian5_matvec_cuda(x: np.ndarray, y: np.ndarray, n: int, k: int, N: int, BT: np.ndarray, deg: np.ndarray):
  # tid = cuda.grid(1)
  tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  ps = cuda.local.array(shape=(7,), dtype=int64)
  if tid < N:
    k_boundary_cuda(tid, k - 1, n, BT, ps)
    a,b,c,d,e,f,g = ps[0], ps[1], ps[2], ps[3], ps[4], ps[5], ps[6]
    cuda.atomic.add(y, a, x[c] + x[e] + x[g] - (x[b] + x[d] + x[f]))
    cuda.atomic.add(y, b, x[d] + x[f] - (x[a] + x[c] + x[e] + x[g]))
    cuda.atomic.add(y, c, x[a] + x[e] + x[g] - (x[b] + x[d] + x[f]))
    cuda.atomic.add(y, d, x[b] + x[f] - (x[a] + x[c] + x[e] + x[g]))
    cuda.atomic.add(y, e, x[a] + x[c] + x[g] - (x[b] + x[d] + x[f]))
    cuda.atomic.add(y, f, x[b] + x[d] - (x[a] + x[c] + x[e] + x[g]))
    cuda.atomic.add(y, g, x[a] + x[c] + x[e] - (x[b] + x[d] + x[f]))

@cuda.jit(device=True)
def searchsorted_cuda(a: np.ndarray, v: np.ndarray):
  """ Searches for index locations of all values 'v' in 'a' via binary search, modifying 'v' in-place """
  n_bins = a.size
  left: int = 0
  right: int = n_bins-1
  # out = cuda.local.array(shape=(v.size,), dtype=int64)
  for i, x in enumerate(v):
    while left < right:
      m: int = left + (right - left) // 2
      m = 0 if m < 0 else min(m, n_bins - 1)
      if a[m] < x:
        left = m + 1
      else:
        right = m
    v[i] = right
    left = right
    right = n_bins - 1

@cuda.jit
def sp_precompute_deg_cuda(n: int, k: int, S: np.ndarray, F: np.ndarray, BT: np.ndarray, deg: np.ndarray) -> np.ndarray:
  tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  ps = cuda.local.array(shape=(8,), dtype=int64)
  N = S.size
  if tid < N:
    k_boundary_cuda(n, S[tid], k - 1, BT, ps)
    searchsorted_cuda(F, ps)
    deg[searchsorted_cuda(F, ps[:k])] += 1

@cuda.jit
def sp_laplacian1_matvec_cuda(
    x: np.ndarray, y: np.ndarray, 
    S: np.ndarray, F: np.ndarray, 
    n: int, k: int, BT: np.ndarray
):
  tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  ps = cuda.local.array(shape=(3,), dtype=int64)
  N = S.size
  if tid < N:
    k_boundary_cuda(S[tid], k - 1, n, BT, ps)
    searchsorted_cuda(F, ps)
    i, j, q = ps[0], ps[1], ps[2]
    cuda.atomic.add(y, i, x[q] - x[j])
    cuda.atomic.add(y, j, -(x[j] + x[q]))
    cuda.atomic.add(y, q, x[i] - x[j])
    