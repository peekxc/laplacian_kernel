import numpy as np
import numba as nb 

@nb.jit(nopython=True)
def construct_flag_dense(N: int, dim: int, n: int, eps: float, weights: np.ndarray, BT: np.ndarray, S: np.ndarray, cc: np.ndarray):
  """Constructs d-simplices of a dense flag complex up to 'eps', optionally discarding apparent pairs."""
  tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  if tid < N:
    w = flag_weight_2(tid, dim, n, weights, BT)
    if w <= eps:
      cx = cuda.atomic.inc(cc, 0, N)
      S[cx] = tid