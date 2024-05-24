
# import _comb_laplacian
from .operators import LaplacianFull, LaplacianSparse, flag_simplices

def compile_laplacians(gpu: bool = False):
  if gpu: 
    from . import laplacian_gpu
    return laplacian_gpu
  else: 
    from . import laplacian_cpu
    return laplacian_cpu

def compile_filtrations(gpu: bool = False):
  if gpu: 
    from . import filtration_gpu
    return filtration_gpu
  else:
    from . import filtration_cpu  
    return filtration_cpu