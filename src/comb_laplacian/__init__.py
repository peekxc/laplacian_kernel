
# import _comb_laplacian
from .operators import LaplacianFull, LaplacianSparse

# def compile_laplacians(gpu: bool = False):
#   if gpu: 
#     from laplacian_gpu import *
#   else: 
#     from operators import Lapl