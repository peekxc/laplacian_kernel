# %% 
from operator import ge
from tempfile import TemporaryFile
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

## To generate the data
# np.random.seed(1234)
X = sample_torus_tube(7500, seed=1234)
# D = squareform(pdist(X))
# con_radius = np.max(minimum_spanning_tree(D).data / 2.0)
# enc_radius = np.min(D.max(axis=1))
# r = con_radius + 0.15 * (enc_radius - con_radius)
# G = neighbor_graph_ball(X, radius=r, weighted=True)

# from pbsig.vis import figure_dgm
# from ripser import ripser
# from scipy.spatial.distance import pdist, squareform
# np.sort(np.diff(ripser(squareform(pdist(X[landmarks(X, 100)])**(3.1)), maxdim=1, distance_matrix=True)['dgms'][1], axis=1).flatten())[-5:]
# np.sort(np.diff(ripser(X[landmarks(X, 250)], maxdim=1)['dgms'][1], axis=1).flatten())[-5:]

## Preload the geodesic distances
DSP = np.load("/Users/mpiekenbrock/laplacian_kernel/data/nbg_torus7500_geodesics.npz")['arr_0']

# %% 
def time_ripser(
  X: np.ndarray, p: int, ratio: float = 1.0, threshold: float = None, 
  ripser_path: str = "~/ripser", 
  verbose: bool = False, 
  profile: str = None, 
  dryrun: bool = False
):
  import os
  import pathlib
  import re
  import subprocess
  from tempfile import NamedTemporaryFile
  if X.ndim == 2 and X.shape[0] == X.shape[1] and np.allclose(X.diagonal(),0):
    d_format = "distance"
    d_suffix = "distance_matrix"
  elif X.ndim == 1:
    d_format = "upper"
    d_suffix = "upper_distance_matrix"
    X = squareform(X)[np.newaxis,:]
  elif X.ndim == 2:
    d_format = "point"
    d_suffix = ""
  else:
    raise ValueError("Invalid input")
  if verbose: 
    print(f"Detected input format {d_format}")

  ## Validate the path to ripser
  ripser_path = "~/ripser"
  ripser_path = os.path.expanduser(ripser_path)
  ripser_path = pathlib.Path(os.path.join(ripser_path, "ripser")).resolve()
  assert ripser_path.exists(), "Did not find file 'ripser'. Did you supply a valid 'ripser_path'?"
  ripser_path_str = str(ripser_path.absolute())

  ## Write the input to a temporary file
  output_tf = NamedTemporaryFile(suffix=d_suffix)
  np.savetxt(output_tf, X, delimiter=",")

  ## Build the command
  cmd = ["/usr/bin/time", "-lp"] if profile else []
  cmd += [ripser_path_str, output_tf.name]
  cmd += ([f"--dim {p}"] + [f"--ratio {ratio}"] + ["--format distance"])
  cmd += [f"--threshold {2*er}"] if threshold is not None else []
  if verbose: 
    print(f"Running command: {' '.join(cmd)}")
  
  ## Run the command
  if dryrun: return ' '.join(cmd)
  out = subprocess.run(' '.join(cmd), shell=True, capture_output=True, check=True)
  
  ## Decode the output
  dgm_info = out.stdout.decode('ascii').split('\n')
  perf_info = out.stderr.decode('ascii').split('\n')

  # for d in range(p+1):
  #   f'persistence intervals in dim {d}:'

  tim_pattern = re.compile(r"(\w+)\s*(\d+\.\d+)")
  mem_pattern = re.compile(r"\s*(\d+)\s*(.*)")
  profile = {}
  profile |= dict([tim_pattern.findall(s)[0] for s in perf_info[:3]])
  profile |= dict([reversed(mem_pattern.findall(s)[0]) for s in perf_info[3:] if len(s) > 0])
  profile = { k : float(v) for i, (k,v) in enumerate(profile.items()) }
  return dgm_info, profile

def profile_ripser_memray(
  X: np.ndarray, p: int, threshold: float = None, 
  prefix: str = "", 
  suffix: str = None
):
  import subprocess
  assert X.ndim == 2, "Only 2-d inputs accepted"
  from ripser import ripser
  from memray import Tracker
  # output_profile = open(out, 'wb') if out is not None else NamedTemporaryFile(suffix=".bin")
  profile_base = f"ripser_{len(X)}_{p}" + ("_" + suffix if suffix is not None else "")
  output_prof = profile_base + ".bin" 
  is_dist = X.shape[0] == X.shape[1] and np.allclose(X.diagonal(),0)
  with Tracker(prefix + output_prof, native_traces=True):  
    res = ripser(X, maxdim=p, thresh=threshold, distance_matrix=is_dist)  
  # subprocess.run(f"memray transform -o {prefix + profile_base + '.csv'} csv {prefix + output_prof}", shell=True, capture_output=True, check=True)
  subprocess.run(f"memray stats --json -o {prefix + profile_base + '.json'} {prefix + output_prof}", shell=True, capture_output=True, check=True)
  return res['dgms']

# %% Torus benchmarks with memray
for n in (np.arange(26, 51) * 100):
  perm = landmarks(DSP, n, radii=False)
  DM = DSP[perm,:][:,perm]
  er = 0.5 * np.min(DM.max(axis=1))
  profile_ripser_memray(DM, p=1, threshold=2*er, prefix="benchmarks/torus/ripser/", suffix="torus")
  # dgm, bench = time_ripser(DM, p=1, ratio=1.0, verbose=True)
  # benchmarks[n] = bench
  print(n)


# import pickle
# # with open('benchmark_torus_ripser.pickle', 'wb') as handle:
# #   pickle.dump(benchmarks, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('benchmark_torus_ripser.pickle', 'rb') as f:
#   benchmarks = pickle.load(f)


from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
output_notebook()

time_ripser = np.array([(k,v['real']) for k,v in benchmarks.items()])
memory_ripser = np.array([(k,v['maximum resident set size']) for k,v in benchmarks.items()])
pages_ripser = np.array([(k,v['page reclaims']) for k,v in benchmarks.items()])

p = figure(width=350, height=250)
p.scatter(*time_ripser.T)
q = figure(width=350, height=250)
q.scatter(*memory_ripser.T)
r = figure(width=350, height=250)
r.scatter(*pages_ripser.T)
show(row(p,q,r))


# %% 
from comb_laplacian import flag_simplices, LaplacianSparse

def profile_laplacian_memray(
  weights: np.ndarray, p: int, threshold: float = None, 
  prefix: str = "", 
  suffix: str = None
):
  import subprocess
  from memray import Tracker
  from combin import inverse_choose
  assert weights.ndim == 1 
  n = inverse_choose(len(weights), 2)
  
  ## Configure file outputs
  profile_base = f"laplacian_{n}_{p}" + ("_" + suffix if suffix is not None else "")
  output_prof = profile_base + ".bin" 

  ## First construct the complex;
  F = flag_simplices(weights, p=p, eps=threshold, discard_ap=False, verbose=True, shortcut=False)
  S = flag_simplices(weights, p=p+1, eps=threshold, discard_ap=True, verbose=True, shortcut=False)
  F = np.sort(F)
  with Tracker(prefix + output_prof, native_traces=True):  
    L = LaplacianSparse(S=S, F=F, n=n, k=p+2, precompute_deg=True, gpu=False)
  
  subprocess.run(f"memray stats --json -o {prefix + profile_base + '.json'} {prefix + output_prof}", shell=True, capture_output=True, check=True)
  return True

p = 1
# (np.arange(1, 51) * 100)
for n in (4000 + np.arange(0,11)*100):
  perm = landmarks(DSP, n, radii=False)
  DM = DSP[perm,:][:,perm]
  er = 0.5 * np.min(DM.max(axis=1))
  w = squareform(DM)
  profile_laplacian_memray(w, p=p, threshold=2*er, prefix="benchmarks/torus/laplacian/", suffix="torus")


# p = 1
# # benchmarks_flag = {}
# for n in (np.arange(1, 51) * 100):
#   perm = landmarks(DSP, n, radii=False)
#   DM = DSP[perm,:][:,perm]
#   er = 0.5 * np.min(DM.max(axis=1))
#   w = squareform(DM)
#   F = flag_simplices(w, p=p, eps=2*er, discard_ap=False, verbose=True, shortcut=False)
#   S = flag_simplices(w, p=p+1, eps=2*er, discard_ap=True, verbose=True, shortcut=False)
#   F = np.sort(F)
#   profile_base = f"laplacian_{len(DM)}_{p}" + ("_" + suffix if suffix is not None else "")
#   output_prof = profile_base + ".bin" 
#   with Tracker():
#     LaplacianSparse(S=S, F=F, n=len(DM), k=p+2, gpu=False)
#   # nbytes = F.nbytes + S.nbytes + (n+1)*(p+1)*8 + (8 * 4) + 96 + 2*F.nbytes
#   # benchmarks_flag[n] = nbytes
#   print(n)

import pickle
# with open('benchmark_torus_laplacian.pickle', 'wb') as handle:
#   pickle.dump(benchmarks_flag, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('benchmark_torus_laplacian.pickle', 'rb') as f:
  benchmarks_flag = pickle.load(f)

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
output_notebook()

p = figure(width=350, height=250)
p.scatter(*np.array(list(benchmarks_flag.items())).T)
show(p)
show(row(p,q,r))

# %% Memray results 
import json
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
from bokeh.models import LogTicker, LogAxis, Range1d
output_notebook()

base_path = "/Users/mpiekenbrock/laplacian_kernel/benchmarks/torus/"

N = np.arange(1,44) * 100
watermark_laplacian = np.array([json.load(open(base_path + f"laplacian/laplacian_{n}_1_torus.json", 'r'))['metadata']['peak_memory'] for n in N])
watermark_ripser = np.array([json.load(open(base_path + f"ripser/ripser_{n}_1_torus.json", 'r'))['metadata']['peak_memory'] for n in N])

p = figure(
  width=500, height=350, y_axis_type="log",
  title="Peak Memory Usage (Torus, Rips H1 persistence)"
)
p.y_range = Range1d(10**4, 10**11)
p.title.align = "center"
p.line(N, watermark_laplacian, color='red', line_width=1.5, legend_label="1-Laplacian (R)")
p.scatter(N, watermark_laplacian, color='red', line_width=0.5, line_color='black')
p.line(N, watermark_ripser, color='blue', line_width=1.5, legend_label="Ripser (Z2)")
p.scatter(N, watermark_ripser, color='blue', line_width=0.5, line_color='black')
p.extra_y_ranges = {"prefix": p.y_range }
right_axis = LogAxis(y_range_name="prefix", axis_label="SI Unit")
right_axis.minor_tick_line_alpha = 0.0
p.add_layout(right_axis, 'right')
# p.toolbar_location = None
p.yaxis[0].axis_label = "Bytes (log-scale)"
p.yaxis[1].axis_label = "Bytes (SI Unit)"
p.yaxis[1].major_label_overrides = {10**4: '10 KB', 10**5: '100 KB', 10**6: '1 MB', 10**7: '10 MB', 10**8: '100 MB', 10**9: '1 GB', 10**10: '10 GB', 10**11: '100 GB'}
# p.yaxis[0].ticker.base = 2
p.legend.location = "bottom_right"
p.legend.title = "Method"
p.xaxis.axis_label = "Number of vertices"
show(p)




#  ripser /var/folders/0l/b3dbb2_d2bb4y3wbbfk0wt_80000gn/T/tmpvuvn8tet.upper_distance_matrix --dim 1 --threshold 2.3701083660125732 --format upper-distance >/dev/null
# /usr/bin/time -lp ripser ../data/nbg_torus500_geodesics.lower_distance_matrix --dim 1 --threshold 2.3701083660125732 --format upper-distance >/dev/null


# %% 
from comb_laplacian import LaplacianSparse, flag_simplices
from primate.trace import hutch
from scipy.sparse.linalg import eigsh
X = sample_torus_tube(7500, seed=1234)
D = pdist(X[landmarks(X, 500)])**3.0
n = 500
er = 0.5*np.min(squareform(D).max(axis=1))
p = 1
F = flag_simplices(D, p=p, eps=2*er, discard_ap=False, verbose=True, shortcut=False)
S = flag_simplices(D, p=p+1, eps=2*er, discard_ap=True, verbose=True, shortcut=False)
F = np.sort(F)

def get_peak_memory(fun):
  import json
  from memray import Tracker 
  from tempfile import NamedTemporaryFile
  import subprocess
  with NamedTemporaryFile() as fn: 
    pass
  with Tracker(fn.name, native_traces=True):  
    fun()
  subprocess.run(f"memray stats --json -o {fn.name + '.json'} {fn.name}", shell=True, capture_output=True, check=True)
  return json.load(open(fn.name + '.json', 'r'))['metadata']['peak_memory'], fn.name


def init_laplacian():
  L = LaplacianSparse(S=S, F=F, n=n, k=p+2, gpu=False)
  tr_est = hutch(L, maxiter=130, deg=50, orth=30, ncv=30, num_threads=1)
  # x = np.zeros(100000)

get_peak_memory(init_laplacian)


M = L.tocoo().todense()
M_rank = np.linalg.matrix_rank(M)

M_rank / L.shape[0]

top_ew = eigsh(M, k = 1, which='LM', return_eigenvectors=False)
ew = np.linalg.eigh(M)[0]
tol = M.shape[0] * top_ew * np.finfo(M.dtype).eps 
gap = np.min(ew[ew > tol])


vol = np.sum(L.deg)
gap_est = 0.5 * (2/vol)**2
tol = 166.227 * L.shape[0] * np.finfo(L.dtype).eps
hutch(M, fun="smoothstep", deg=40, maxiter=5, a=0.1*gap, b=gap) / L.shape[0]



def profile_laplacian_memray_inline(
  weights: np.ndarray, p: int, threshold: float = None, 
  prefix: str = "", 
  suffix: str = None
):
  from tempfile import NamedTemporaryFile
  import subprocess
  from memray import Tracker
  from combin import inverse_choose
  assert weights.ndim == 1 
  n = inverse_choose(len(weights), 2)

  with NamedTemporaryFile() as fn: 
    pass
  fn.name
  
  ## First construct the complex;
  F = flag_simplices(weights, p=p, eps=threshold, discard_ap=False, verbose=True, shortcut=False)
  S = flag_simplices(weights, p=p+1, eps=threshold, discard_ap=True, verbose=True, shortcut=False)
  F = np.sort(F)
  with Tracker(fn.name, native_traces=True):  
    L = LaplacianSparse(S=S, F=F, n=n, k=p+2, precompute_deg=True, gpu=False)
  
  subprocess.run(f"memray stats --json -o {prefix + profile_base + '.json'} {prefix + output_prof}", shell=True, capture_output=True, check=True)
  return True


u = 2500
p = figure(width=300, height=250, y_axis_type="log")
p.line(np.linspace(0, u), np.linspace(0, u), color='red')
p.line(np.linspace(0, u), np.linspace(0, u)**2, color='blue')
# p.line(np.linspace(0, 5000), 0.001 * np.linspace(0, 15000)**3, color='blue')
show(p)