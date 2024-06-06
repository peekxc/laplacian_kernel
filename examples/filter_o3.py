import numpy as np
from math import comb 
from comb_laplacian import LaplacianFull, LaplacianSparse, compile_filtrations, compile_laplacians, flag_simplices
from scipy.spatial.distance import pdist, cdist, squareform
from combin import rank_to_comb, comb_to_rank
from landmark import landmarks
from comb_laplacian.filtration_cpu import apparent_blocker, filter_flag_ap

# X = np.loadtxt("https://raw.githubusercontent.com/Ripser/ripser-benchmark/master/o3_4096.txt")
X = np.loadtext("/Users/mpiekenbrock/laplacian_kernel/data/o3_4096.txt")
w = pdist(X)
n = len(X)

from comb_laplacian.operators import rips_laplacian
X = X[landmarks(X, 30)]
rips_laplacian(X, p=1, discard_ap=True, sparse=False)



## Read file in
o3_4 = np.fromfile('/Users/mpiekenbrock/laplacian_kernel/o3_4simplices_4k.bin', dtype=np.int64)
o3_4.sort()
o3_4 = np.flip(o3_4).copy()
# o3_4 = o3_4[np.argsort(-o3_4)]

from array import array
s_neg = array('Q')
block_size = 256
S_out = -np.ones(block_size, dtype=np.int64)
n_blocks = (o3_4.size // block_size) + 1
increment = (int(o3_4.size * 0.01))
for ii in range(n_blocks):
  s = ii * block_size
  e = min((ii + 1) * block_size, o3_4.size)
  S_out.fill(-1)
  filter_flag_ap(o3_4[s:e], dim=4, n=n, eps=1.4, weights=w, BT=BT, S_out=S_out) 
  ns1 = len(s_neg)
  s_neg.extend(S_out[S_out != -1])
  ns2 = len(s_neg)
  if ii % 1500 == 0: 
    print(f"Block: {ii}/{n_blocks}, fraction: {(ii/n_blocks):.3f}, added: {ns2-ns1}")

# np.array(s_neg, dtype=np.int64).tofile("o3_4simplices_ap.bin")
s_neg = np.array(s_neg, dtype=np.int64)
s_neg_ap = np.fromfile('o3_4simplices_ap.bin', dtype=np.int64)


## Obtain the 2 and 3 simplices 
## ...



from comb_laplacian import LaplacianSparse
F = np.fromfile('o3_3simplices_4k.bin', dtype=np.int64)
F.sort()
S = np.fromfile('o3_4simplices_4k_ap.bin', dtype=np.int64)
L = LaplacianSparse(S, F, n=len(X), k=5)




