
# import urllib.request

# laplacian_url = "https://raw.githubusercontent.com/peekxc/laplacian_kernel/main/laplacian.cu"
# f = urllib.request.urlopen(laplacian_url)
# LP_cu = f.read()
# LP_cu.decode("utf-8").replace("\\n", "\n")

import numpy as np
import claplacian
from math import comb

n, k = 10, 3
N = comb(n, k)
x = np.ones(comb(n, k-1))
y = np.zeros(len(x))
claplacian._claplacian.laplacian1_matvec(k,n,N,x,y)


import numba



