import numpy as np
from math import comb 
from comb_laplacian import LaplacianFull, LaplacianSparse, compile_filtrations, compile_laplacians, flag_simplices
from scipy.spatial.distance import pdist, cdist, squareform
from combin import rank_to_comb
from landmark import landmarks


X = np.loadtxt("https://raw.githubusercontent.com/Ripser/ripser-benchmark/master/o3_4096.txt")[:50]
# X = X[landmarks(X, 1024)]
w = pdist(X)
n = len(X)


# from comb_laplacian import compile_filtrations
# fm = compile_filtrations()
# BT = np.array([[int(comb(ni, ki)) for ni in range(n+1)] for ki in range(k+2)]).astype(np.int64)
# [fm.flag_weight(ii, 1, n, w, BT) for ii in range(15)]

from simplextree import SimplexTree
st = SimplexTree([[i] for i in np.arange(len(X))])
# st.insert(rank_to_comb(np.flatnonzero(w <= 1.8), k=2, n=n, order='lex'))
st.insert(rank_to_comb(np.flatnonzero(w <= 1.4), k=2, n=n, order='lex'))
st.expand(4)

# flag_simplices(w, p=1, eps=1.8, discard_ap=False, n_blocks=1, verbose=True)

comb(4096,4) // 2**19

## In summary: just use some software to try to build  it
n, p = len(X), 3
S = flag_simplices(w, p=p+1, eps=1.4, discard_ap=True, n_blocks=2056, verbose=True, shortcut=True)
F = flag_simplices(w, p=p, eps=1.4, discard_ap=False, n_blocks=2**19, verbose=True, shortcut=False)


# F = np.sort(F)
# L = LaplacianSparse(S,F,n,k)
import gudhi
rips_complex = gudhi.RipsComplex(
  points=X,
  max_edge_length=1.4
)
st = rips_complex.create_simplex_tree()

import numba as nb

def my_block(dim: int):
  @nb.jit(nopython=True)
  def _block(s: list):
    return len(s) == dim + 1
  return _block


st.expansion_with_blocker(2, my_block(dim=2))
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