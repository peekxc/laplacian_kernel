
import numpy as np
from landmark import landmarks
from comb_laplacian import flag_simplices
from scipy.spatial.distance import squareform
from line_profiler import LineProfiler, profile

DSP = np.load("/Users/mpiekenbrock/laplacian_kernel/data/nbg_torus7500_geodesics.npz")['arr_0']

n = 100
perm = landmarks(DSP, n, radii=False)
DM = DSP[perm,:][:,perm]
er = 0.5 * np.min(DM.max(axis=1))
weights = squareform(DM)

# profiler = LineProfiler()
# profiler.add_function(flag_simplices)
# profiler.print_stats()

p = 1
F = flag_simplices(weights, p=p, eps=2*er, discard_ap=False, verbose=False, shortcut=False)
S = flag_simplices(weights, p=p+1, eps=2*er, discard_ap=True, verbose=False, shortcut=False)

from comb_laplacian import LaplacianSparse
L = LaplacianSparse(S=S, F=F, n=n, k=p+2)
M = L.tocoo()
M = M / eigsh(M, k=1)[0]
# np.linalg.matrix_rank(M.todense())

# 3835
from primate.trace import hutch
lanczos_quad
# hutch(M, maxiter=250, fun="smoothstep", a=1e-4, b=1e-2, verbose=True, deg=40)
import optuna
study = optuna.create_study()

def rank_obj(trial):
  a = trial.suggest_float('a', 1e-10, 1e-1)
  b = trial.suggest_float('b', a, 1e-1)
  tr_est = hutch(M, maxiter=250, fun="smoothstep", a=a, b=b, verbose=True, deg=40)
  return (tr_est - 3835)**2 

study.optimize(rank_obj, n_trials=1500)

study.best_trial.params
hutch(M, maxiter=250, fun="smoothstep", verbose=True, deg=40, **study.best_trial.params, plot=True)
hutch(M, maxiter=250, fun="smoothstep", verbose=True, deg=40, **study.best_trial.params)


from scipy.sparse.linalg import eigsh
gap = eigsh(M, k=1, which='LM', return_eigenvectors=False, sigma=0.0005)[0]
eigsh(M, k=1, which='LM', return_eigenvectors=False)

true_ew = np.linalg.eigh(M.todense())[0]

tol = np.max(M.shape) * np.finfo(M.dtype).eps
np.sum(true_ew >= tol)

a = np.max(true_ew[true_ew <= tol])
b = np.min(true_ew[true_ew >= tol])
hutch(M, maxiter=250, fun="smoothstep", verbose=True, deg=40, a=a, b=b)
hutch(M, maxiter=250, fun="smoothstep", verbose=True, deg=40, a=a, b=100*b, plot=True)

hutch(M, maxiter=250, fun="smoothstep", verbose=True, deg=40, a=1500000*a, b=gap)


from primate.quadrature import sl_gauss
nodes, weights = sl_gauss(M, n=1, deg=40, orth=5).T
c = (M.shape[0] / 1)
c * np.sum(nodes * weights) 

from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from scipy.stats import gaussian_kde
output_notebook()

node_p = np.argsort(nodes)

p = figure(width=300, height=250)
# p.scatter(nodes, weights)
p.line(nodes[node_p], weights[node_p])
show(p)

kde = gaussian_kde(nodes, bw_method=0.05)
node_dens = kde.evaluate(nodes)

p = figure(width=300, height=250)
p.line(nodes[node_p], node_dens[node_p])
show(p)

kde = gaussian_kde(np.abs(true_ew), bw_method=0.0010)
ew_dens = kde.evaluate(np.abs(true_ew))
ew_p = np.argsort(np.abs(true_ew))

p = figure(width=300, height=250)
p.line(np.abs(true_ew)[ew_p], ew_dens[ew_p])
show(p)

tol = np.finfo(M.dtype).eps*np.max(M.shape)
np.min(true_ew[true_ew >= tol])
np.histogram(true_ew[true_ew < tol])

np.sum(true_ew >= 0.00112)

denom = 1.0 / (1 + np.exp(-10*(np.maximum(0.0, np.abs(true_ew) - gap))))
result = 2.0*(denom - 0.5)
M.shape[0] - np.sum(result == 0.0)

np.sum(result)


def exp_weight(x: np.ndarray, epsilon: float, alpha: float = 10.0):
  denom = 1.0 / (1 + np.exp(-alpha*(np.maximum(0.0, np.abs(x) - epsilon))))
  return 2.0*(denom - 0.5)

  nodes * weights

from primate.special import softsign
ss = lambda x, q: np.maximum(softsign(x, q), 0.0)
rank_est = np.array([c * np.sum(weights * ss(nodes, int(q))) for q in np.geomspace(1, 15000, 100)])

p = figure(width=350, height=200)
p.line(np.arange(M.shape[0]), rank_est)
show(p)


M.trace()




# from primate.functional import numrank
# numrank(M, gap=1e-6, verbose=True, psd=True)


