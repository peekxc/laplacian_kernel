import numpy as np
from comb_laplacian import LaplacianSparse

X = np.loadtxt("https://raw.githubusercontent.com/Ripser/ripser-benchmark/master/o3_4096.txt")
F = np.fromfile('/Users/mpiekenbrock/laplacian_kernel/o3_3simplices_4k.bin', dtype=np.int64)
S = np.fromfile('/Users/mpiekenbrock/laplacian_kernel/o3_4simplices_4k_ap.bin', dtype=np.int64)
L = LaplacianSparse(S, F, n=len(X), k=5)

y = L @ np.ones(L.shape[1])
