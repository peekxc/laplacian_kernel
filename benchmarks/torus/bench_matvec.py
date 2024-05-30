import numpy as np
from ripser import ripser
from array import array
from scipy.sparse.csgraph import floyd_warshall, minimum_spanning_tree
from scipy.spatial.distance import pdist, cdist, squareform
from set_cover.covers import neighbor_graph_ball, neighbor_graph_knn, neighbor_graph_del
from scipy.sparse.csgraph import floyd_warshall
from landmark import landmarks
from ripser import ripser
from comb_laplacian.sampler import sample_torus_tube
from comb_laplacian import flag_simplices

DSP = np.load("/Users/mpiekenbrock/laplacian_kernel/data/nbg_torus7500_geodesics.npz")['arr_0']
p = 1
n = 200
# for n in (np.arange(1, 51) * 100):
perm = landmarks(DSP, n, radii=False)
DM = DSP[perm,:][:,perm]
er = 0.5 * np.min(DM.max(axis=1))
w = squareform(DM)

F = flag_simplices(w, p=p, eps=2*er, discard_ap=False, verbose=True, shortcut=False)
S = flag_simplices(w, p=p+1, eps=2*er, discard_ap=True, verbose=True, shortcut=False)
F = np.sort(F)
L = LaplacianSparse(S=S, F=F, n=n, k=p+2, precompute_deg=False, gpu=False)

x = np.arange(L.shape[1])
import timeit
timeit.timeit(lambda: L @ x, number=30) / 30

from ripser import ripser
timeit.timeit(lambda: ripser(DM, maxdim=1, thresh=2*er, distance_matrix=True), number=30) / 30