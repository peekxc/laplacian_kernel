from typing import Callable
from math import comb, sqrt
from numba import float32, int64, uint64, void, cuda
import numpy as np
import numba as nb
import cupy as cp

# @cuda.jit(tuple(int64, int64), device=True)
# def rank_to_comb_lex2(r: int, n: int):
#   i: int64 = n - 2 - int64(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5)
#   j: int64 = r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
#   return (i,j)

## For flag complexes
# @cuda.jit(float32(int64, int64, int64, float32[:], int64[:,:]), device=True)
@cuda.jit(device=True)
def simplex_weight_1(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = cuda.local.array(shape=(2,), dtype=int64)
  rank_to_comb_colex(simplex, n, 2, BT, labels)
  return weights[comb_to_rank_lex2(labels[0], labels[1], n)]

# @cuda.jit(float32(int64, int64, int64, float32[:], int64[:,:]), device=True)
@cuda.jit(device=True)
def simplex_weight_2(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = cuda.local.array(shape=(3,), dtype=int64)
  rank_to_comb_colex(simplex, n, 3, BT, labels)

  s_weight = weights[comb_to_rank_lex2(labels[0], labels[1], n)]
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[0], labels[2], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[1], labels[2], n)])
  return s_weight

@cuda.jit(float32(int64, int64, int64, float32[:], int64[:,:]), device=True)
def simplex_weight_3(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = cuda.local.array(shape=(4,), dtype=int64)
  rank_to_comb_colex(simplex, n, 4, BT, labels)

  s_weight = weights[comb_to_rank_lex2(labels[0], labels[1], n)]
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[0], labels[2], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[0], labels[3], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[1], labels[2], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[1], labels[3], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[2], labels[3], n)])
  return s_weight

## For lower-star
# @cuda.jit(device=True)
# def simplex_weight_1(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
#   labels = cuda.local.array(shape=(2,), dtype=int64)
#   rank_to_comb_colex(simplex, n, 2, BT, labels)
#   return max(weights[labels[0]], weights[labels[1]])

# @cuda.jit(device=True)
# def simplex_weight_2(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
#   labels = cuda.local.array(shape=(3,), dtype=int64)
#   rank_to_comb_colex(simplex, n, 3, BT, labels)
#   return max(weights[labels[0]], weights[labels[1]], weights[labels[2]])

# @cuda.jit(device=True)
# def simplex_weight_3(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
#   labels = cuda.local.array(shape=(4,), dtype=int64)
#   rank_to_comb_colex(simplex, n, 4, BT, labels)
#   return max(weights[labels[0]], weights[labels[1]], weights[labels[2]], weights[labels[3]])

@cuda.jit(int64(int64, int64, int64, float32[:], int64[:,:]), device=True)
def cofacet_search_2(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray) -> int:

  ## Get current simplex weight
  weight = simplex_weight_2(simplex, dim, n, weights, BT)

  ## Cofacet search
  # k_cofacets = cuda.local.array(shape=(8-(3),), dtype=int64)
  # k_coboundary_cuda(simplex, dim, n, BT, k_cofacets)
  idx_below: int64 = simplex
  idx_above: int64 = 0
  j: int64 = n - 1
  k: int64 = dim + 1
  c: int64 = 0
  zero_cofacet = -1
  while j >= k:
    while BT[k][j] <= idx_below:
      idx_below -= BT[k][j]
      idx_above += BT[k+1][j]
      j -= 1
      k -= 1
    cofacet_index = idx_above + BT[k+1][j] + idx_below
    j -= 1

    ## If a cofacet with the same weight is found, return it
    cofacet_weight = simplex_weight_3(cofacet_index, dim+1, n, weights, BT)
    if cofacet_weight == weight or c > 256:
      zero_cofacet = cofacet_index
      break
    c += 1
  return zero_cofacet

@cuda.jit(int64(int64, int64, int64, float32[:], int64[:,:]), device=True)
def zero_facet_2(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray) -> int64:
  '''Given a dim-dimensional simplex, find its lexicographically minimal facet with identical simplex weight'''
  c_weight: float = simplex_weight_2(simplex, dim, n, weights, BT)
  zero_facet: int = -1

  ## Compute the boundary of the simplex
  k_facets = cuda.local.array(shape=(3,), dtype=int64)
  k_boundary_cuda(simplex, dim, n, BT, k_facets)

  ## Return the first simplex with identical weight if it exists
  for c_facet in k_facets:
    facet_weight = simplex_weight_1(c_facet, dim-1, n, weights, BT)
    # print(f"{c_facet} => {facet_weight:.5f}")
    if facet_weight == c_weight:
      zero_facet = c_facet
      break
  return zero_facet

@cuda.jit(int64(int64, int64, int64, float32[:], int64[:,:]), device=True)
def zero_facet_3(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray) -> int64:
  '''Given a dim-dimensional simplex, find its lexicographically minimal facet with identical simplex weight'''
  c_weight: float32 = simplex_weight_3(simplex, dim, n, weights, BT)
  zero_facet: int64 = (-1)

  ## Compute the boundary of the simplex
  k_facets = cuda.local.array(shape=(4,), dtype=int64)
  k_boundary_cuda(simplex, dim, n, BT, k_facets)

  ## Return the first simplex with identical weight if it exists
  for c_facet in k_facets:
    facet_weight = simplex_weight_2(c_facet, dim-1, n, weights, BT)
    # print(f"{c_facet} => {facet_weight:.5f}")
    if facet_weight == c_weight:
      zero_facet = c_facet
      break
  return zero_facet

@cuda.jit(void(int64, int64, int64, float32, float32[:], int64[:,:], int64[:], uint64[:]))
def construct_flag_dense_ap(N: int, dim: int, n: int, eps: float, weights: np.ndarray, BT: np.ndarray, S: np.ndarray, cc: np.ndarray):
  """Constructs d-simplices of a dense flag complex up to 'eps', optionally discarding apparent pairs."""
  tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  ## Normally we would find AP's; here we only keep non-apparent d-simplices
  if tid < N:
    w = simplex_weight_2(tid, dim, n, weights, BT)
    if w <= eps:
      c = cofacet_search_2(tid, dim, n, weights, BT)
      z = zero_facet_3(c, dim+1, n, weights, BT)
      if c == -1 or tid != z:
        # cx = cuda.atomic.add(cc, 0, 1)
        cx = cuda.atomic.inc(cc, 0, N)
        S[cx] = tid