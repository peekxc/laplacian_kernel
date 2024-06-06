from typing import Callable
from numbers import Number
from math import comb
from scipy.sparse.linalg import LinearOperator

import numpy as np
import numba as nb
from set_cover.covers import squareform

# https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size
def sizeof_fmt(num, suffix="B"):
  for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
    if abs(num) < 1024.0:
      return f"{num:3.1f}{unit}{suffix}"
    num /= 1024.0
  return f"{num:.1f}Yi{suffix}"

class LaplacianABC():
  def __init__(self, n: int, k: int, N: int, M: int, gpu: bool = False) -> None:
    assert k >= 2, "k must be at least 2"
    self.gpu = gpu
    self.n = n # num vertices
    self.k = k # dim + 1
    self.N = N ## num of (k-2)-simplices, comb(n,k-1)
    self.M = M ## num of (k-1)-simplices, comb(n,k)
    self.BT = np.array([[comb(ni, ki) for ni in range(n+1)] for ki in range(k+2)]).astype(np.int64)
    assert np.all(self.BT >= 0), "Binomial coefficient calculation failed"
    self.shape = (self.N, self.N)
    self.dtype = np.dtype('float32')

    ## Define the array module
    if not gpu: 
      self.xp = np 
    else: 
      import cupy as cp
      self.xp = cp

    ## Constants / cache
    self._BT = self.xp.asarray(self.BT)
    self._y = self.xp.zeros(self.N, dtype=np.float32)
    self.nbytes = self._y.nbytes + self._BT.nbytes + (8 * 4) + 96
  
  def __repr__(self) -> str:
    # runtime_bytes = 3 * ((comb(self.n,self.k-1) * 4) / 1024**3) + (self.n*(self.k+2) * 8)
    msg = f"Up-Laplacian ({self.N} / {self.M}) - ({self.k-2}/{self.k-1})-simplices \n"
    msg += f"n: {self.n}, k: {self.k}, "
    msg += (f"Memory usage: {sizeof_fmt(self.nbytes)} \n")
    if hasattr(self, "threadsperblock"):
      msg += f"threads-per-block: {self.threadsperblock}, blocks-per-thread: {self.blockspergrid}\n"
    return msg

  def tocoo(self):
    from scipy.sparse import coo_array
    from array import array
    x = np.zeros(self.shape[1])
    I, J, D = array('I'), array('I'), array('f')
    for j in range(len(x)):
      x[j-1] = 0
      x[j] = 1
      y = self._matvec(x).copy()
      r = np.flatnonzero(y)
      I.extend(r)
      J.extend(np.repeat(j, len(r)))
      D.extend(y[r])
    if self.gpu:
      import cupyx
      from cupyx.scipy.sparse import coo_matrix as coo
    else: 
      from scipy.sparse import coo_array as coo
    I,J,D = self.xp.asarray(I), self.xp.asarray(J), self.xp.asarray(D)
    return coo((D, (I,J)), shape=self.shape, dtype=np.float32)
  
class LaplacianFull(LinearOperator, LaplacianABC):
  """Up-Laplacian operator on the full n-simplex."""
  def __init__(self, n: int, k: int, gpu: bool = False, threadsperblock: int = 32, n_kernels: int = 1):
    assert k >= 2, "k must be at least 2"
    M,N = comb(n,k), comb(n,k-1)
    super(LinearOperator, self).__init__(n, k, N, M, gpu) # calls LaplacianABC / excludes LinearOperator
    self.deg = self.xp.ones(self.N, dtype=np.float32) * (n - k + 1)
    self.nbytes += self.deg.nbytes
    if not gpu:
      from .laplacian_cpu import laplacian0_matvec, laplacian1_matvec, laplacian2_matvec, laplacian3_matvec, laplacian4_matvec, laplacian5_matvec
      if k == 2:
        self.launch_config = laplacian0_matvec
      elif k == 3:
        self.launch_config = laplacian1_matvec
      elif k == 4:
        self.launch_config = laplacian2_matvec
      elif k == 5:
        self.launch_config = laplacian3_matvec
      elif k == 6:
        self.launch_config = laplacian4_matvec
      elif k == 7:
        self.launch_config = laplacian5_matvec
      else:
        raise ValueError("invalid k")
    else:
      from .laplacian_gpu import laplacian1_matvec_cuda, laplacian2_matvec_cuda, laplacian3_matvec_cuda, laplacian4_matvec_cuda, laplacian5_matvec_cuda
      self.threadsperblock = threadsperblock
      self.blockspergrid = ((self.M + (self.threadsperblock - 1)) // threadsperblock) + 1
      self.n_kernels = n_kernels
      # self._y = cupyx.zeros_pinned(self.N, dtype=np.float32)
      self._y = self.xp.zeros(self.N, dtype=np.float32)
      if k == 3:
        self.launch_config = laplacian1_matvec_cuda[self.blockspergrid, self.threadsperblock]
      elif k == 4:
        self.launch_config = laplacian2_matvec_cuda[self.blockspergrid, self.threadsperblock]
      elif k == 5:
        self.launch_config = laplacian3_matvec_cuda[self.blockspergrid, self.threadsperblock]
      elif k == 6:
        self.launch_config = laplacian4_matvec_cuda[self.blockspergrid, self.threadsperblock]
      elif k == 7:
        self.launch_config = laplacian5_matvec_cuda[self.blockspergrid, self.threadsperblock]
      else:
        raise ValueError("invalid k")

  def _matvec(self, x: np.ndarray) -> np.ndarray:
    x = self.xp.asarray(x).flatten()
    y = self.xp.asarray(self._y)
    self.xp.multiply(x, self.deg, y)
    # cp.cuda.stream.get_current_stream().synchronize()
    self.launch_config(x, y, self.n, self.k, self.M, self._BT)
    return self.xp.asnumpy(y) if self.gpu else y
    # return y.get()
  
  def __repr__(self) -> str:
    return LaplacianABC.__repr__(self)

## Laplacian Sparse Operator
class LaplacianSparse(LinearOperator, LaplacianABC):
  """Up-Laplacian operator on a sparse complex."""
  def __init__(self, 
    S: np.ndarray, F: np.ndarray, 
    n: int, k: int, 
    precompute_deg: bool = True,
    gpu: bool = False, threadsperblock: int = 32, n_kernels: int = 1, 
  ):
    from .laplacian_cpu import sp_precompute_deg
    assert k >= 2, "k must be at least 2"
    M,N = len(S), len(F)
    super(LinearOperator, self).__init__(int(n), int(k), N, M, gpu) # calls LaplacianABC / excludes LinearOperator

    ## Precompute degree + allocate device memory
    self.S = self.xp.asarray(S)
    self.F = self.xp.asarray(F)
    self.F.sort() ## Required for searchsorted
    self.deg = self.xp.asarray(sp_precompute_deg(n,k,S,F,self.BT)) if precompute_deg else self.xp.zeros(N, dtype=np.int64)
    self.nbytes += (self.deg.nbytes + self.S.nbytes + self.F.nbytes)

    if not gpu:
      from .laplacian_cpu import sp_laplacian1_matvec, sp_laplacian2_matvec, sp_laplacian3_matvec, sp_laplacian4_matvec, sp_laplacian5_matvec
      if k == 3:
        self.launch_config = sp_laplacian1_matvec
      elif k == 4:
        self.launch_config = sp_laplacian2_matvec
      elif k == 5:
        self.launch_config = sp_laplacian3_matvec
      elif k == 6:
        self.launch_config = sp_laplacian4_matvec
      elif k == 7:
        self.launch_config = sp_laplacian5_matvec
      else:
        raise ValueError(f"Invalid k = {k}; should be in [3,4,5,6,7]")
    else:
      from .laplacian_gpu import sp_laplacian1_matvec_cuda, sp_laplacian2_matvec_cuda, sp_laplacian3_matvec_cuda, sp_laplacian4_matvec_cuda, sp_laplacian5_matvec_cuda
      self.threadsperblock = threadsperblock
      self.blockspergrid = ((self.M + (self.threadsperblock - 1)) // threadsperblock) + 1
      self.n_kernels = n_kernels
      if k == 3:
        self.launch_config = sp_laplacian1_matvec_cuda[self.blockspergrid, self.threadsperblock]
      if k == 4:
        self.launch_config = sp_laplacian2_matvec_cuda[self.blockspergrid, self.threadsperblock]
      if k == 5:
        self.launch_config = sp_laplacian3_matvec_cuda[self.blockspergrid, self.threadsperblock]
      if k == 6:
        self.launch_config = sp_laplacian4_matvec_cuda[self.blockspergrid, self.threadsperblock]
      if k == 7:
        self.launch_config = sp_laplacian5_matvec_cuda[self.blockspergrid, self.threadsperblock]
      else:
        raise ValueError(f"Invalid k = {k}; should be in [3]")

  def _matvec(self, x: np.ndarray) -> np.ndarray:
    assert len(x) == len(self.deg), f"Invalid dimensions for 'x'; should be {len(self.deg)}"
    x = self.xp.asarray(x).flatten()
    y = self.xp.asarray(self._y) # Note: by default this is zero-copy 
    self.xp.multiply(x, self.deg, y)
    self.launch_config(x, y, self.S, self.F, self.n, self.k, self._BT)
    # cp.cuda.stream.get_current_stream().synchronize()
    return self.xp.asnumpy(y) if self.gpu else y

  def __repr__(self) -> str:
    return LaplacianABC.__repr__(self)
  
  # def tocoo(self):
  #   return LaplacianABC.tocoo(self)

## Blocked approach to constructing the simplices of a flag filtration up to some threshold
def flag_simplices(weights: np.ndarray, p: int, eps: float, n_blocks: int = 'auto', discard_ap: bool = True, verbose: bool = False, shortcut: bool = True):
  from .filtration_cpu import construct_flag_dense_ap, construct_flag_dense
  from array import array
  from combin import inverse_choose
  assert isinstance(weights, np.ndarray), "Weights must be a numpy array"
  if weights.ndim == 2 and weights.shape[0] == weights.shape[1]:
    weights = squareform(weights)
  n, k = int(inverse_choose(weights.size, 2, exact=True)), p+1
  N, M = comb(n,k-1), comb(n,k)
  BT = np.array([[int(comb(ni, ki)) for ni in range(n+1)] for ki in range(k+2)]).astype(np.int64)

  ## Choose the construction function
  construct_f = construct_flag_dense if not discard_ap else construct_flag_dense_ap

  ## Use the sqrt-heuristic to pick the number of blocks
  shift = max((int(np.sqrt(n))-1).bit_length() + 1, 1) if n > 20 else 0
  n_blocks = 1 << shift if n_blocks == 'auto' else int(n_blocks)
  ns_per_block = (M // n_blocks) + 1
  assert (ns_per_block * n_blocks) >= M, "Bad block configuration"

  ## Construct the simplices as signed int64's 
  S = array('q')
  S_out = -1 * np.ones(ns_per_block, dtype=np.int64)
  if verbose: 
    print(f"enumerating {n_blocks} blocks", flush=True)
  for bi in reversed(range(n_blocks)):
    offset = bi*ns_per_block
    # n_launches = min(M, offset + ns_per_block) - offset
    n_launches = min(ns_per_block, np.abs(M - offset))
    S_out.fill(-1)
    construct_f(n_launches, dim=p, n=n, eps=eps, weights=weights, BT=BT, S=S_out, offset=offset)
    new_s = S_out[S_out != -1]
    if verbose: 
      print(f"block: {n_blocks-bi}/{n_blocks}, {p}-simplices: {new_s.size} / {n_launches}", flush=True)
    S.extend(new_s)
    if shortcut and len(new_s) == 0: 
      break ## Shortcut that is unclear why it works
  del S_out 
  return np.asarray(S, dtype=np.int64)


def rips_laplacian(
  X: np.ndarray, p: int, radius: float = "default", sparse: bool = True, 
  discard_ap: bool = True, sp_mat: bool = False, 
  gpu: bool = False
):
  """Constructs a sparse Laplacian operator from a point cloud suitable for p-dimensional homology computations.
  
  This function constructs the (p/p+1)-simplices needed to compute p-dimensional homology via a Laplacian operator. 
  Optionally, 
  0-Laplacian <=> vertex/edge (graph) Laplacian 
  1-Laplacian <=> edge/triangle (mesh) Laplacian

  """
  from scipy.spatial.distance import pdist, cdist, squareform
  weights = pdist(X)
  if radius == "default":
    radius = 0.5 * np.min(np.max(squareform(weights), axis=1)) # enclosing radius 
  assert isinstance(radius, Number), "Radius must be a number or 'default'."
  
  ## Constants
  n,diam = len(X), 2*radius
  
  ## Constructs the flag simplices
  if not sparse: 
    F = flag_simplices(weights, p=p, eps=diam, discard_ap=False, verbose=False, shortcut=False)
    S = flag_simplices(weights, p=p+1, eps=diam, discard_ap=True, verbose=False, shortcut=False)
    F.sort()
    L = LaplacianSparse(S=S, F=F, n=n, k=p+2, precompute_deg=True, gpu=False)
    return L
  else: 
    from combin import comb_to_rank
    from comb_laplacian.filtration_cpu import apparent_blocker
    import gudhi
    DM = squareform(weights)
    rips_complex = gudhi.RipsComplex(distance_matrix=DM, max_edge_length=diam)
    st = rips_complex.create_simplex_tree()
    if discard_ap: 
      blocker_fun = apparent_blocker(maxdim=p+1, n=n, eps=diam, weights=weights)
      blocker_fun(np.arange(p+1)) # compiles it
      st.expansion_with_blocker(p+1, blocker_fun)
    else:
      st.expansion(p+1)
    S_vert = np.array([s for s, fv in st.get_simplices() if len(s) == (p+2)])
    F_vert = np.array([s for s, fv in st.get_simplices() if len(s) == (p+1)])
    S = comb_to_rank(S_vert, order='colex', n=n, k=p+2)
    F = comb_to_rank(F_vert, order='colex', n=n, k=p+1)
    F.sort()
    L = LaplacianSparse(S=S, F=F, n=n, k=p+2, precompute_deg=True, gpu=False)
    return L




## From: https://stackoverflow.com/questions/3160699/python-progress-bar
# def progressbar(it, count=None, prefix="", size=60, out=sys.stdout, f: Callable = None, newline: bool = True): # Python3.6+
#   count = len(it) if count == None else count 
#   f = (lambda j: "") if f is None else f
#   def show(j):
#     x = int(size*j/count)
#     print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}" + f(j), end='\r', file=out, flush=True)
#   show(0)
#   for i, item in enumerate(it):
#     yield item
#     show(i+1)
#   print("\n" if newline else "", flush=True, file=out)
