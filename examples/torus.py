import numpy as np 
from array import array
from scipy.sparse.csgraph import floyd_warshall, minimum_spanning_tree
from scipy.spatial.distance import pdist, cdist, squareform
from set_cover.covers import neighbor_graph_ball, neighbor_graph_knn, neighbor_graph_del
from scipy.sparse.csgraph import floyd_warshall
from landmark import landmarks
from ripser import ripser
from comb_laplacian.sampler import sample_torus_tube

# from bokeh.plotting import figure, show
# from bokeh.io import output_notebook
# output_notebook()

np.random.seed(1234)
X = sample_torus_tube(7500, seed=1234)
X[:5,]
# p = figure(width=300, height=300)
# p.scatter(*X[:,:2].T, size=5)
# show(p)

D = squareform(pdist(X))
con_radius = np.max(minimum_spanning_tree(D).data / 2.0)
enc_radius = np.min(D.max(axis=1))
r = con_radius + 0.15 * (enc_radius - con_radius)
G = neighbor_graph_ball(X, radius=r, weighted=True)
# DSP = np.load("/Users/mpiekenbrock/laplacian_kernel/data/nbg_torus7500_geodesics.npz")['arr_0']
DSP = floyd_warshall(G, directed=False) # 9m for 7500 
# np.savetxt("/Users/mpiekenbrock/laplacian_kernel/data/nbg_torus7500_geodesics.lower_distance_matrix", squareform(DSP))
# np.savez("/Users/mpiekenbrock/laplacian_kernel/nbg_torus7500_geodesics.npz", DSP)
# G = neighbor_graph_knn(X, k=40, weighted=True)
# G = neighbor_graph_del(X, weighted=True)



## Get ripser 
/Users/mpiekenbrock/laplacian_kernel/data/nbg_torus7500_geodesics.npz
perm = landmarks(DSP, 50, radii=False)
DM = DSP[perm,:][:,perm]
er = 0.5 * np.min(DM.max(axis=1))
np.savetxt(f"/Users/mpiekenbrock/laplacian_kernel/data/nbg_torus{len(perm)}_geodesics.lower_distance_matrix", squareform(DM))

# /usr/bin/time -lp ripser data/nbg_torus500_geodesics.lower_distance_matrix --dim 1 --threshold 2.3701083660125732 --format upper-distance >/dev/null
ripser(DM, thresh=2*er, maxdim=1, distance_matrix=True)

# import timeit149986568
# timeit.timeit(lambda: ripser(DM, thresh=2*er, maxdim=2, distance_matrix=True), number = 1)
# timeit.timeit(lambda: ripser(DM, maxdim=2, distance_matrix=True), number = 1)


## Sparse complex
# import gudhi
# 
# rips_complex = gudhi.RipsComplex(
#   distance_matrix=DM,
#   max_edge_length=2*er
# )
# st = rips_complex.create_simplex_tree()
# st.expansion(3)

from comb_laplacian import flag_simplices
from scipy.spatial.distance import squareform
from comb_laplacian.filtration_cpu import construct_flag_dense_ap
from math import comb

# w = squareform(DM)
# BT = np.array([[int(comb(ni, ki)) for ni in range(len(X)+1)] for ki in range(6)]).astype(np.int64)
59809792 / (1024**2)

# import timeit
# timeit.timeit(lambda: flag_simplices(DM, 2, eps=2*er, verbose=False, shortcut=False), number=10)
S = flag_simplices(DM, 2, eps=2*er, verbose=True, n_blocks=512, shortcut=False)





import timeit
def bench(b: int):
  S_out = -1 * np.ones(b, dtype=np.int64)
  S_out.fill(-1)
  construct_flag_dense_ap(b, dim=2, n=len(DM), eps=2*er, weights=w, BT=BT, S=S_out, offset=0)

for d in 2**np.arange(16):
  time_per = timeit.timeit(lambda: bench(d), number=100) / d 
  print(f"{d}: time per: {time_per:.5f}, fraction: {d / comb(len(X), 3):.6f}")

wut = np.load("/Users/mpiekenbrock/Downloads/nbg_torus1500_S_dim2.npy")
len(wut)
wut.nbytes / (1024**3)

# from scipy.sparse import save_npz, load_npz
# np.savez("torus7500.npz", X)
# # np.savez("")
# save_npz("nbg_torus7500.npz", G)
# np.savez("nbg_torus7500_geodesics.npz", DSP)

# X = np.load("torus7500.npz")['arr_0']
# G = load_npz("nbg_torus7500.npz")
# DSP = np.load("nbg_torus7500_geodesics.npz")['arr_0']
# perm = landmarks(X, 7500)

# G_sparse = G[perm[:800],:][:,perm[:800]]



import line_profiler
from comb_laplacian.operators import flag_simplices
from comb_laplacian.filtration_cpu import construct_flag_dense_ap, zero_facet_flag, zero_cofacet_flag, flag_weight
from comb_laplacian.combinatorial_cpu import zero_cofacet_flag


profile = line_profiler.LineProfiler()
profile.add_function(flag_simplices)
profile.add_function(construct_flag_dense_ap)
# profile.add_function(zero_facet_flag)
# profile.add_function(zero_cofacet_flag)
# profile.add_function(flag_weight)
profile.enable_by_count()
S = flag_simplices(DM, 2, eps=2*er, n_blocks=8192, verbose=True, shortcut=False)

profile.print_stats()


# from comb_laplacian.filtration_cpu import construct_flag_dense, construct_flag_dense_ap
# construct_flag_dense_ap()

