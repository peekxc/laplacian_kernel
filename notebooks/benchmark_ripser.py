import numpy as np
from math import comb 
from comb_laplacian import LaplacianFull, LaplacianSparse, compile_filtrations, compile_laplacians, flag_simplices
from scipy.spatial.distance import pdist, cdist, squareform
from combin import rank_to_comb, comb_to_rank
import comb_laplacian
from landmark import landmarks
import gudhi
from comb_laplacian.filtration_cpu import apparent_blocker, filter_flag_ap

X = np.loadtxt("https://raw.githubusercontent.com/Ripser/ripser-benchmark/master/o3_4096.txt")
# X = X[landmarks(X, 1500)]
w = pdist(X)
n = len(X)
BT = np.array([[int(comb(ni, ki)) for ni in range(n+1)] for ki in range(7)]).astype(np.int64)
eps = 1.4 # np.median(w)

## Sparse complex
rips_complex = gudhi.RipsComplex(
  points=X,
  max_edge_length=eps
)
st = rips_complex.create_simplex_tree()
st.expansion(3)
# st.expansion_with_blocker(2, lambda s: False)

o3_3simplices = np.array([s for s,sf in st.get_simplices() if len(s) == 4], dtype=np.int64)
from combin import comb_to_rank
o3_3_simplices = np.array([comb_to_rank(s, order='colex', n=len(X)) for s in o3_3simplices], dtype=np.int64)
o3_3_simplices = np.array(o3_3_simplices, dtype=np.int64)
o3_3_simplices.tofile("o3_3simplices_4k.bin")


C.sort(axis=1)
C = np.fliplr(C) if colex_order else C
C = np.array(C, order='K', copy=True)

blocker_fun = apparent_blocker(maxdim=2, n=len(X), eps=eps, weights=w, BT=BT)
blocker_fun(np.arange(5)) # compiles it
st.expansion_with_blocker(2, blocker_fun)

st.num_simplices() # 207 
len([s for s,sf in st.get_simplices() if len(s) == 4])

len(flag_simplices(w, p=2, eps=eps, discard_ap=False, n_blocks=1, verbose=True, shortcut=False))
len(flag_simplices(w, p=2, eps=eps, discard_ap=True, n_blocks=6, verbose=True, shortcut=False))

# from comb_laplacian.filtration_cpu import flag_weight
# flag_weights = np.array([flag_weight(i, dim=2, n=n, weights=w, BT=BT) for i in range(comb(n,3))]) 
# np.sum(flag_weights <= eps)

# triangles = flag_simplices(w, p=2, eps=eps, discard_ap=False, n_blocks=1, verbose=True, shortcut=False)
# triangle_labels = rank_to_comb(triangles, k=3, n=n, order='colex')

# from comb_laplacian.filtration_cpu import apparent_blocker
# blocker_fun = apparent_blocker(maxdim=2, n=len(X), eps=eps, weights=w, BT=BT)
# blocker_fun(np.arange(5)) # compiles it
# wut = [blocker_fun(t) for t in triangle_labels]

# np.sum([not blocker_fun(t) for t in triangle_labels])

# neg_triangles = flag_simplices(w, p=2, eps=eps, discard_ap=True, n_blocks=1, verbose=True, shortcut=False)
# neg_triangles_labels = rank_to_comb(neg_triangles, k=3, n=n, order='colex')

# # np.all([blocker_fun(t) for t in neg_triangles_labels])
# blocker_fun([8,14,19])

## In summary: just use some software to try to build  it
n, p = len(X), 2
S = flag_simplices(w, p=p+1, eps=2.4, discard_ap=False, n_blocks=8, verbose=True, shortcut=False)
F = flag_simplices(w, p=p, eps=1.4, discard_ap=False, n_blocks=2**19, verbose=True, shortcut=False)


# k = 4
# BT = np.array([[int(comb(ni, ki)) for ni in range(n+1)] for ki in range(k+2)]).astype(np.int64)
# comb_to_rank_colex([0,1,2], n=n, BT=BT)
# np.all(np.unique([comb_to_rank_colex(list(s), n=n, BT=BT) for s in combinations(range(10), 3)]) == np.arange(comb(10,3)))

from combin import comb_to_rank
comb_to_rank([0,1,2], n=n, order='colex')

# from comb_laplacian import compile_filtrations
# fm = compile_filtrations()
# BT = np.array([[int(comb(ni, ki)) for ni in range(n+1)] for ki in range(k+2)]).astype(np.int64)
# [fm.flag_weight(ii, 1, n, w, BT) for ii in range(15)]

from simplextree import SimplexTree
st = SimplexTree([[i] for i in np.arange(len(X))])
# st.insert(rank_to_comb(np.flatnonzero(w <= 1.8), k=2, n=n, order='lex'))
st.insert(rank_to_comb(np.flatnonzero(w <= 1.4), k=2, n=n, order='lex'))
st.expand(4)


# F = np.sort(F)
# L = LaplacianSparse(S,F,n,k)
import gudhi
rips_complex = gudhi.RipsComplex(
  points=X,
  max_edge_length=1.4
)
st = rips_complex.create_simplex_tree()

from comb_laplacian.filtration_cpu import apparent_blocker
blocker_fun = apparent_blocker(maxdim=2, n=len(X), eps=2.4, weights=w, BT=BT)
blocker_fun(np.arange(5)) # compiles it
st.expansion_with_blocker(2, blocker_fun)
st.num_simplices()

# st.expansion(4)

from itertools import islice
from array import array
from combin import comb_to_rank
ns = st.num_simplices()
sranks = array('Q')
cc = 0
for s, fs in st.get_simplices():
  if len(s) == 5:
    sranks.append(comb_to_rank(s, order='colex', n=len(X)))
    cc += 1
    if (cc % (ns // 100)) == 0:
      print(cc)

np.savetxt('o3_4simplices.out', sranks, delimiter=',')

filter_flag_ap

# blocker_fun = apparent_blocker(maxdim=4, n=len(X), eps=1.4, weights=w, BT=BT)
# with open('o3_4simplices.out') as fin:
#   cc = 0
#   for line in fin:
#     r = int(float(line[:-1]))
#     simplex = rank_to_comb(r, k=5, n=n, order='colex')
#     if not blocker_fun(list(simplex)):
#       s_neg.append(r)
#     cc += 1
#     if cc % 5000 == 0: 
#       print(cc % 5000)
#       # break
      


o3_4 = np.loadtxt('o3_4simplices.out', dtype=np.int64, delimiter=',')