from typing import Callable
from math import comb
from scipy.sparse.linalg import LinearOperator

import numpy as np
import numba as nb

# https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size
def sizeof_fmt(num, suffix="B"):
  for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
    if abs(num) < 1024.0:
      return f"{num:3.1f}{unit}{suffix}"
    num /= 1024.0
  return f"{num:.1f}Yi{suffix}"

class LaplacianABC():
  def __init__(self, n: int, k: int, gpu: bool = False) -> None:
    assert k >= 2, "k must be at least 2"
    self.gpu = gpu
    self.n = n # num vertices
    self.k = k # dim + 1
    self.N = comb(n,k-1)
    self.M = comb(n,k)
    self.BT = np.array([[comb(ni, ki) for ni in range(n+1)] for ki in range(k+2)]).astype(np.int64)
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
    msg = f"Up-Laplacian ({self.N} / {self.M}) - ({self.k-1}/{self.k})-simplices \n"
    msg += f"n: {self.n}, k: {self.k}, "
    msg += (f"Memory usage: {sizeof_fmt(self.nbytes)} \n")
    if hasattr(self, "threadsperblock"):
      msg += f"threads-per-block: {self.threadsperblock}, blocks-per-thread: {self.blockspergrid}\n"
    return msg

class LaplacianFull(LinearOperator, LaplacianABC):
  """Up-Laplacian operator on the full n-simplex."""
  def __init__(self, n: int, k: int, gpu: bool = False, threadsperblock: int = 32, n_kernels: int = 1):
    assert k >= 2, "k must be at least 2"
    super(LinearOperator, self).__init__(n, k, gpu) # calls LaplacianABC / excludes LinearOperator
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
    return coo((D, (I,J)), shape=self.shape, dtype=np.float32)

## Laplacian Sparse Operator
class LaplacianSparse(LinearOperator, LaplacianABC):
  """Up-Laplacian operator on a sparse complex."""
  def __init__(self, 
    S: np.ndarray, F: np.ndarray, 
    n: int, k: int, 
    gpu: bool = False, threadsperblock: int = 32, n_kernels: int = 1
  ):
    from .laplacian_cpu import sp_precompute_deg
    assert k >= 2, "k must be at least 2"
    super(LinearOperator, self).__init__(n, k, gpu) # calls LaplacianABC / excludes LinearOperator
    
    ## Precompute degree + allocate device memory
    self.deg = self.xp.asarray(sp_precompute_deg(n,k,S,F,self.BT))
    self.S = self.xp.asarray(S)
    self.F = self.xp.asarray(F)
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
      from .laplacian_gpu import sp_laplacian1_matvec_cuda, sp_laplacian2_matvec, sp_laplacian3_matvec, sp_laplacian4_matvec, sp_laplacian5_matvec
      self.threadsperblock = threadsperblock
      self.blockspergrid = ((self.M + (self.threadsperblock - 1)) // threadsperblock) + 1
      self.n_kernels = n_kernels
      if k == 3:
        self.launch_config = sp_laplacian1_matvec_cuda[self.blockspergrid, self.threadsperblock]
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