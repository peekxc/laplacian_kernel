import numpy as np
import cupy as cp
from primate.trace import hutch
from LaplacianOp import LaplacianFull

L = LaplacianFull(n=250, k=3, gpu=True)
print(L)
x = cp.arange(L.N, dtype=np.float32)
y = L.matvec(x)
print(y)

# from primate.operator import matrix_function 
# M = matrix_function(L, fun="numrank")
# print(M.matvec(x))

import line_profiler
profiler = line_profiler.LineProfiler()

profiler.add_function(hutch)
profiler.add_function(L.matvec)
profiler.enable_by_count()
hutch(L, fun="numrank", deg=5, maxiter=1500)
profiler.print_stats(output_unit=1e-3, stripzeros=True)


# wut = hutch(L, fun="numrank", maxiter=1500)
# print(wut)