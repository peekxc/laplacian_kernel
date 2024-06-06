from numba.np.ufunc import parallel
import numpy as np
import numba as nb 
from numba import int64, float32, float64, uint64, void, prange
from .combinatorial_cpu import comb_to_rank_lex2, rank_to_comb_colex, comb_to_rank_colex, k_boundary_cpu, k_coboundary_cpu, do_bounds_check

@nb.jit(nopython=True)
def flag_weight_1(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = np.zeros(2, dtype=int64)
  rank_to_comb_colex(simplex, n, 2, BT, labels)
  return weights[comb_to_rank_lex2(labels[0], labels[1], n)]

# @cuda.jit(float32(int64, int64, int64, float32[:], int64[:,:]), device=True)
@nb.jit(nopython=True)
def flag_weight_2(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = np.zeros(3, dtype=int64)
  rank_to_comb_colex(simplex, n, 3, BT, labels)

  s_weight = weights[comb_to_rank_lex2(labels[0], labels[1], n)]
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[0], labels[2], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[1], labels[2], n)])
  return s_weight

@nb.jit(nopython=True)
def flag_weight_3(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = np.zeros(4, dtype=int64)
  rank_to_comb_colex(simplex, n, 4, BT, labels)

  s_weight = weights[comb_to_rank_lex2(labels[0], labels[1], n)]
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[0], labels[2], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[0], labels[3], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[1], labels[2], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[1], labels[3], n)])
  s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[2], labels[3], n)])
  return s_weight

@nb.jit(nopython=True)
def flag_weight(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = np.zeros(8, dtype=np.int64)
  rank_to_comb_colex(simplex, n, dim+1, BT, labels)
  s_weight = -np.inf
  for i in range(dim+1):
    for j in range(i+1, dim+1):
      r = comb_to_rank_lex2(labels[i], labels[j], n)
      s_weight = max(s_weight, weights[r])
      # print(f"s: {simplex}, ({i},{j}): li: {labels[i]}, lj: {labels[j]}, n:{n}, r:{r}, wr: {weights[r]:.5f}")
  return s_weight

@nb.jit(nopython=True)
def pc_weight(simplex: int, dim: int, n: int, points: np.ndarray, BT: np.ndarray):
  labels = np.zeros(8, dtype=np.int64)
  rank_to_comb_colex(simplex, n, dim+1, BT, labels)
  s_weight = -np.inf
  for i in range(dim+1):
    for j in range(i+1, dim+1):
      s_weight = max(s_weight, np.sum((points[labels[i]] - points[labels[j]])**2))
  return np.sqrt(s_weight)

@nb.jit(nopython=True)
def flag_weight_labels(simplex: list, n: int, weights: np.ndarray) -> float:
  k = len(simplex)
  s_weight = -np.inf
  for i in range(k):
    for j in range(i+1, k):
      r = comb_to_rank_lex2(simplex[i], simplex[j], n)
      s_weight = max(s_weight, weights[r])
  return s_weight

## For lower-star
@nb.jit(nopython=True)
def star_weight_1(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = np.zeros(2, dtype=int64)
  rank_to_comb_colex(simplex, n, 2, BT, labels)
  return max(weights[labels[0]], weights[labels[1]])

@nb.jit(nopython=True)
def star_weight_2(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = np.zeros(3, dtype=int64)
  rank_to_comb_colex(simplex, n, 3, BT, labels)
  return max(weights[labels[0]], weights[labels[1]], weights[labels[2]])

@nb.jit(nopython=True)
def star_weight_3(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = np.zeros(4, dtype=int64)
  rank_to_comb_colex(simplex, n, 4, BT, labels)
  return max(weights[labels[0]], weights[labels[1]], weights[labels[2]], weights[labels[3]])

@nb.jit(nopython=True)
def star_weight(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  labels = np.zeros(8, dtype=int64)
  rank_to_comb_colex(simplex, n, dim+1, BT, labels)
  return np.max(weights[labels[:(dim+1)]])

@nb.jit(nopython=True)
def cofacet_search_2(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray) -> int:

  ## Get current simplex weight
  weight = flag_weight_2(simplex, dim, n, weights, BT)

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
    cofacet_weight = flag_weight_3(cofacet_index, dim+1, n, weights, BT)
    if cofacet_weight == weight or c > 256:
      zero_cofacet = cofacet_index
      break
    c += 1
  return zero_cofacet

# @nb.jit(nopython=True)
# def zero_facet_2(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray) -> int64:
#   '''Given a dim-dimensional simplex, find its lexicographically minimal facet with identical simplex weight'''
#   c_weight: float = flag_weight(simplex, dim, n, weights, BT)
#   zero_facet: int = -1

#   ## Compute the boundary of the simplex
#   k_facets = np.zeros(3, dtype=int64)
#   k_boundary_cpu(simplex, dim, n, BT, k_facets)

#   ## Return the first simplex with identical weight if it exists
#   for c_facet in k_facets:
#     facet_weight = flag_weight_1(c_facet, dim-1, n, weights, BT)
#     # print(f"{c_facet} => {facet_weight:.5f}")
#     if facet_weight == c_weight:
#       zero_facet = c_facet
#       break
#   return zero_facet


@nb.jit(nopython=True, boundscheck=do_bounds_check)
def zero_cofacet_flag(simplex: int, dim: int, weight: float, n: int, weights: np.ndarray, BT: np.ndarray, max_c: int = 4096):
  # weight = flag_weight(simplex, dim, n, weights, BT)
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
    j -= 1
    cofacet_weight = flag_weight(cofacet_index, dim+1, n, weights, BT)
    if cofacet_weight == weight or c > max_c:
      return cofacet_index
  return -1

@nb.jit(nopython=True)
def zero_facet_flag(simplex: int, dim: int, weight: float, n: int, weights: np.ndarray, BT: np.ndarray) -> int64:
  '''Given a dim-dimensional simplex, find its lexicographically minimal facet with identical simplex weight'''
  # c_weight: float = flag_weight(simplex, dim, n, weights, BT)

  ## Compute the boundary of the simplex
  k_facets = np.zeros(8, dtype=int64)
  k_boundary_cpu(simplex, dim, n, BT, k_facets)

  ## Return the first simplex with identical weight if it exists
  for c_facet in k_facets:
    facet_weight = flag_weight(c_facet, dim-1, n, weights, BT)
    # print(f"{c_facet} => {facet_weight:.5f}")
    if facet_weight == weight:
      return c_facet
  return -1

# @nb.jit(int64(int64, int64, int64, float32[:], int64[:,:]), nopython=True)
# def zero_facet_3(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray) -> int64:
#   '''Given a dim-dimensional simplex, find its lexicographically minimal facet with identical simplex weight'''
#   c_weight: float32 = flag_weight_3(simplex, dim, n, weights, BT)
#   zero_facet: int64 = (-1)

#   ## Compute the boundary of the simplex
#   k_facets = np.zeros(4, dtype=int64)
#   k_boundary_cpu(simplex, dim, n, BT, k_facets)

#   ## Return the first simplex with identical weight if it exists
#   for c_facet in k_facets:
#     facet_weight = flag_weight_2(c_facet, dim-1, n, weights, BT)
#     # print(f"{c_facet} => {facet_weight:.5f}")
#     if facet_weight == c_weight:
#       zero_facet = c_facet
#       break
#   return zero_facet

def apparent_blocker(maxdim: int, n: int, eps: float, weights: np.ndarray):
  from math import comb
  BT = np.array([[int(comb(ni, ki)) for ni in range(n+1)] for ki in range(maxdim+3)]).astype(np.int64)
  @nb.jit(nopython=True)
  def _block(simplex: list) -> bool:
    dim = len(simplex) - 1
    w = flag_weight_labels(simplex, n, weights)
    if dim < maxdim and w <= eps:
      return False # accept simplex
    elif dim == maxdim and w <= eps: 
      s = comb_to_rank_colex(simplex, BT)
      c = zero_cofacet_flag(s, dim, w, n, weights, BT)
      if c == -1 or s != zero_facet_flag(c, dim+1, w, n, weights, BT):
        # print(f"Accept: {s}, c={c}, z={zero_facet_flag(c, dim+1, w, n, weights, BT)}, w={w:.4f}")
        return False  # accept simplex
    # print(f"Reject: {s}, c={c}, z={zero_facet_flag(c, dim+1, w, n, weights, BT)}, w={w:.4f}")
    return True      # reject simplex
  return _block
  
@nb.jit(nopython = True, parallel=True)
def filter_flag_ap(S: np.ndarray, dim: int, n: int, eps: float, weights: np.ndarray, BT: np.ndarray, S_out: np.ndarray):
  """Constructs d-simplices of a dense flag complex up to 'eps', optionally discarding apparent pairs."""
  ## Normally we would find AP's; here we only keep non-apparent d-simplices
  for tid in prange(S.size):
    s = S[tid]
    w = flag_weight(s, dim, n, weights, BT)
    if w <= eps:
      c = zero_cofacet_flag(s, dim, w, n, weights, BT)
      if c == -1 or s != zero_facet_flag(c, dim+1, w, n, weights, BT):
        S_out[tid] = s # assumes S is large enough 
  

@nb.jit(nopython = True, parallel=True)
def construct_flag_dense_ap(N: int, dim: int, n: int, eps: float, weights: np.ndarray, BT: np.ndarray, S: np.ndarray, offset: int = 0):
  """Constructs d-simplices of a dense flag complex up to 'eps', optionally discarding apparent pairs."""
  ## Normally we would find AP's; here we only keep non-apparent d-simplices
  for tid in prange(N):
    s = offset + tid
    w = flag_weight(s, dim, n, weights, BT)
    if w <= eps:
      c = zero_cofacet_flag(s, dim, w, n, weights, BT)
      if c == -1 or s != zero_facet_flag(c, dim+1, w, n, weights, BT):
        S[tid] = s # assumes S is large enough 
      # print(f"{s}: => c:{c}, w:{w:.4f}, z: {zero_facet_flag(c, dim+1, w, n, weights, BT)}") 
  
@nb.jit(nopython=True, parallel=True)
def construct_flag_dense(N: int, dim: int, n: int, eps: float, weights: np.ndarray, BT: np.ndarray, S: np.ndarray, offset: int = 0):
  """Constructs d-simplices of a dense flag complex up to 'eps'."""
  for tid in prange(N):
    s = offset + tid
    w = flag_weight(s, dim, n, weights, BT)
    if w <= eps:
      S[tid] = s
    # print(f"{s}: => w:{w:.4f}") 

# void(int64, int64, int64, float32, float32[:], int64[:,:], int64[:])
# @nb.jit(nopython = True, parallel=True)
# def construct_flag_dense_ap(N: int, dim: int, n: int, eps: float, weights: np.ndarray, BT: np.ndarray, S: np.ndarray):
#   """Constructs d-simplices of a dense flag complex up to 'eps', optionally discarding apparent pairs."""
#   ## Normally we would find AP's; here we only keep non-apparent d-simplices
#   for tid in prange(N):
#     w = flag_weight_2(tid, dim, n, weights, BT)
#     if w <= eps:
#       c = cofacet_search_2(tid, dim, n, weights, BT)
#       z = zero_facet_3(c, dim+1, n, weights, BT)
#       if c == -1 or tid != z:
#         S[tid] = tid # assumes S is large enough 