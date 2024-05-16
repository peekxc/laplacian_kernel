#!/usr/bin/python
import numpy as np
import numba as nb
import cupy as cp

print("JIT-compiling CPU kernels...")
from laplacian_cpu import *

print("JIT-compiling GPU kernels...")
from laplacian_gpu import *

def predict_GB(n,k):
  return 3 * ((comb(n,k-1) * 4) / 1024**3) + (n*(k+2) * 8) / 1024**3

class LaplacianBenchmark():
  def __init__(self, n: int, k: int, gpu: bool = False, threadsperblock: int = 32, n_kernels: int = 1):
    assert k >= 2, "k must be at least 2"
    self.tm = np if not gpu else cp
    self.n = n # num vertices
    self.k = k # dim + 1
    self.N = comb(n,k-1)
    self.M = comb(n,k)
    self._pred_GB = 3 * ((comb(self.n,self.k-1) * 4) / 1024**3) + (self.n*(self.k+2) * 8) / 1024**3
    # self.x = cp.ones(self.N)
    BT = np.array([[comb(ni, ki) for ni in range(n)] for ki in range(k+2)]).astype(np.int64)


    if not gpu:
      self.mult = np.multiply
      self.deg = np.ones(self.N, dtype=np.float32) * (n - k + 1)
      self.BT = BT
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
      self.mult = cp.multiply
      self.deg = cp.ones(self.N, dtype=np.float32) * (n - k + 1)
      self.BT = cp.array(BT)
      self.threadsperblock = threadsperblock
      self.blockspergrid = ((self.M + (self.threadsperblock - 1)) // threadsperblock) + 1
      self.n_kernels = n_kernels
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

  def __repr__(self) -> str:
    msg = f"Up-Laplacian (({self.N} / {self.M})  ({self.k-1}/{self.k})-simplices) \n"
    msg += f"n: {self.n}, k: {self.k}, deg: {self.deg[:2]}\n"
    msg += (f"Pred memory usage cap: {self._pred_GB:.3f} GB \n")
    if hasattr(self, "threadsperblock"):
      msg += f"threads-per-block: {self.threadsperblock}, blocks-per-thread: {self.blockspergrid}\n"
    return msg

  def __call__(self, x: np.ndarray, y: np.ndarray, offset: int = 0) -> np.ndarray:
    assert len(x) == len(self.deg) and len(y) == len(self.deg), "Invalid dimensions"
    self.mult(x, self.deg, y)
    self.launch_config(x, y, self.n, self.k, self.M, self.BT, self.deg)
    return y

print("--- Creating benchmark ---")
L = LaplacianBenchmark(16, 4, gpu=True, n_kernels=1)
print(L)

## Do a single matvec
# x = L.tm.arange(L.N)
# y = L.tm.zeros(L.N)
# L(x,y)

import timeit

timings = {}
for n in [16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1024, 1448]:
  for k in [3,4,5,6,7]:
    if predict_GB(n,k) > 30:
      continue
    L = LaplacianBenchmark(n, k, gpu=True)
    x = L.tm.arange(L.N)
    y = L.tm.zeros(L.N)
    print((n,k))
    timings[(n,k)] = timeit.repeat(lambda: L(x,y), number=1, repeat=30)
  print(f"Finished: n={n}")

from json import dumps, dump
timings_ = { str(k) : v for k,v in timings.items()}
with open('timings.json', 'w') as fp:
    dump(timings_, fp)