
# import _comb_laplacian
from .operators import LaplacianFull, LaplacianSparse, flag_simplices

# def compile_laplacians(gpu: bool = False):
#   if gpu: 
#     from .laplacian_gpu import *
#   else: 
#     from .laplacian_cpu import *

def compile_filtrations(gpu: bool = False):
  if gpu: 
    from . import filtration_gpu
    return filtration_gpu
  else:
    from . import filtration_cpu  
    return filtration_cpu