
# import _comb_laplacian
from .operators import LaplacianFull, LaplacianSparse

# def compile_laplacians(gpu: bool = False):
#   if gpu: 
#     from .laplacian_gpu import *
#   else: 
#     from .laplacian_cpu import *

def compile_filtrations(gpu: bool = False):
  if gpu: 
    from .filtration_gpu import construct_flag_dense, construct_flag_dense_ap
  else:
    from . import filtration_cpu 
    from .filtration_cpu import construct_flag_dense, construct_flag_dense_ap
    return filtration_cpu # construct_flag_dense, construct_flag_dense_ap


    # N: int, dim: int, n: int, eps: float, weights: np.ndarray, BT: np.ndarray, S: np.ndarray