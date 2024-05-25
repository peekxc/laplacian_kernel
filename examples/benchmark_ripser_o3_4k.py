# import resource
import numpy as np
# import filprofiler
from ripser import ripser

X = np.loadtxt("https://raw.githubusercontent.com/Ripser/ripser-benchmark/master/o3_4096.txt")[:50]
out = ripser(X, maxdim=3, thresh=1.4)

# print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)