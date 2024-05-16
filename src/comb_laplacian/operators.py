from typing import Callable
from math import comb
from scipy.sparse.linalg import LinearOperator

import numpy as np
import numba as nb

class LaplacianFull():
  """Up-Laplacian operator on the full n-simplex."""
  def __init__(self, n: int, k: int, gpu: bool = False, threadsperblock: int = 32, n_kernels: int = 1):
    assert k >= 2, "k must be at least 2"
    self.gpu = gpu
    self.n = n # num vertices
    self.k = k # dim + 1
    self.N = comb(n,k-1)
    self.M = comb(n,k)
    BT = np.array([[comb(ni, ki) for ni in range(n+1)] for ki in range(k+2)]).astype(np.int64)
    self.shape = (self.N, self.N)
    self.dtype = np.dtype('float32')
    if not gpu:
      from .laplacian_cpu import laplacian1_matvec, laplacian2_matvec, laplacian3_matvec, laplacian4_matvec, laplacian5_matvec
      self.xp = np
      self.mult = np.multiply
      self.deg = np.ones(self.N, dtype=np.float32) * (n - k + 1)
      self.BT = BT
      self._y = np.zeros(self.N, dtype=np.float32)
      if k == 3:
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
      import cupy as cp
      self.xp = cp
      self.mult = cp.multiply
      self.deg = cp.ones(self.N, dtype=np.float32) * (n - k + 1)
      self.BT = cp.array(BT)
      self.threadsperblock = threadsperblock
      self.blockspergrid = ((self.M + (self.threadsperblock - 1)) // threadsperblock) + 1
      self.n_kernels = n_kernels
      # self._y = cupyx.zeros_pinned(self.N, dtype=np.float32)
      self._y = cp.zeros(self.N, dtype=np.float32)
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
  
  def predict_GB(self):
    return 3 * ((comb(self.n,self.k-1) * 4) / 1024**3) + (self.n*(self.k+2) * 8) / 1024**3
  
  def __repr__(self) -> str:
    msg = f"Up-Laplacian (({self.N} / {self.M})  ({self.k-1}/{self.k})-simplices) \n"
    msg += f"n: {self.n}, k: {self.k}, deg: {self.deg[:2]}\n"
    msg += (f"Pred memory usage cap: {self.predict_GB():.3f} GB \n")
    if hasattr(self, "threadsperblock"):
      msg += f"threads-per-block: {self.threadsperblock}, blocks-per-thread: {self.blockspergrid}\n"
    return msg

  def matvec(self, x: np.ndarray) -> np.ndarray:
    x = self.xp.asarray(x)
    y = self.xp.asarray(self._y)
    self.mult(x, self.deg, y)
    # cp.cuda.stream.get_current_stream().synchronize()
    self.launch_config(x, y, self.n, self.k, self.M, self.BT, self.deg)
    return self.xp.asnumpy(y) if self.gpu else y
    # return y.get()
  
## Laplacian Sparse Operator
class LaplacianSparse():
  """Up-Laplacian operator on a sparse complex."""
  def __init__(self, 
    S: np.ndarray, F: np.ndarray, 
    n: int, k: int, 
    gpu: bool = False, threadsperblock: int = 32, n_kernels: int = 1
  ):
    from .laplacian_cpu import sp_precompute_deg
    assert k >= 2, "k must be at least 2"
    self.gpu = gpu
    self.n = n # num vertices
    self.k = k # dim + 1
    self.N = comb(n,k-1)
    self.M = comb(n,k)
    
    ## Precompute degree 
    BT = np.array([
      [comb(ni, ki) for ni in range(n)] for ki in range(k+2)]
    ).astype(np.int64)
    deg = sp_precompute_deg(n,k,S,F,BT)

    ## Allocate device arrays
    self.deg = self.tm.asarray(deg)
    self.BT = self.tm.asarray(BT)
    self.shape = (self.N, self.N)
    self.dtype = np.dtype('float32')
    self.S = self.tm.asarray(S, dtype=np.int64) # p simplices (colex indices) 
    self.F = self.tm.asarray(F, dtype=np.int64) # (p-1) simplices (colex indices)   
    self._y = self.tm.zeros(self.N, dtype=np.float32)
    
    if not gpu:
      from .laplacian_cpu import sp_laplacian1_matvec, sp_laplacian2_matvec, sp_laplacian3_matvec, sp_laplacian4_matvec, sp_laplacian5_matvec
      self.mult = np.multiply
      # self.deg = np.ones(self.N, dtype=np.float32) * (n - k + 1)
      # self.deg = sp_precompute_deg(n,k,S,F,self.BT)
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
      import cupy as cp
      self.xp = cp
      self.mult = cp.multiply
      # self.deg = cp.ones(self.N, dtype=np.float32) * (n - k + 1)
      # self.BT = cp.array(BT)
      # self.deg = sp_precompute_deg(n,k,S,F,self.BT)
      # self.deg = self.tm.zeros(self.N)
      self.threadsperblock = threadsperblock
      self.blockspergrid = ((self.M + (self.threadsperblock - 1)) // threadsperblock) + 1
      self.n_kernels = n_kernels
      
      ## Precompute degree
      
      # sp_precompute_deg_cuda[self.blockspergrid, self.threadsperblock](self.n, self.k, self.S, self.F, self.BT, self.deg)
      
      if k == 3:
        self.launch_config = sp_laplacian1_matvec_cuda[self.blockspergrid, self.threadsperblock]
      else:
        raise ValueError("invalid k")
  
  def predict_GB(self):
    return 3 * ((comb(self.n,self.k-1) * 4) / 1024**3) + (self.n*(self.k+2) * 8) / 1024**3
  
  def __repr__(self) -> str:
    msg = f"Up-Laplacian (({self.N} / {self.M})  ({self.k-1}/{self.k})-simplices) \n"
    msg += f"n: {self.n}, k: {self.k}, deg: {self.deg[:2]}\n"
    msg += (f"Pred memory usage cap: {self.predict_GB():.3f} GB \n")
    if hasattr(self, "threadsperblock"):
      msg += f"threads-per-block: {self.threadsperblock}, blocks-per-thread: {self.blockspergrid}\n"
    return msg

  def matvec(self, x: np.ndarray) -> np.ndarray:
    x = self.xp.asarray(x)
    y = self.xp.asarray(self._y)
    self.mult(x, self.deg, y)
    # cp.cuda.stream.get_current_stream().synchronize()
    self.launch_config(x, y, self.S, self.F, self.n, self.k, self.BT)
    # return self.tm.asnumpy(y) if self.gpu else y
    # return y.get() if self.gpu else y
    return self.xp.asnumpy(y) if self.gpu else y
  
  def __call__(self, x: np.ndarray, y: np.ndarray, offset: int = 0) -> np.ndarray:
    assert len(x) == len(self.deg) and len(y) == len(self.deg), "Invalid dimensions"
    self.mult(x, self.deg, y)
    self.launch_config(x, y, self.S, self.F, self.n, self.k, self.M, self.BT)
    return y