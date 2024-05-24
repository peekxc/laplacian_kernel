import numpy as np 
from array import array


def rs_tube_tube(n: int, r: float, b: int = "auto") -> np.ndarray:
  """Rejection sampler for torus"""
  x = array('f')
  C_PI, R_CPI = 2 * np.pi, 1 / np.pi
  b = int(np.sqrt(n)) if b == "auto" else int(b)
  while len(x) < n: 
    theta = np.random.uniform(size=b, low=0, high=C_PI)
    jac_theta = (1.0 + r * np.cos(theta)) / C_PI
    density_threshold = np.random.uniform(size=n, low=0, high=R_PI)
    x.extend(theta[jac_theta > density_threshold])
  return np.array(x)[:n]

## Based on TDAunif package by Cory Brunson
## Which is based on the paper sampling from a manifold
def sample_torus_tube(n: int, ar: int = 2, sd: int = 0, seed = None) -> np.ndarray:
  np.random.seed(seed)
  r = 1.0 / ar
  theta = rs_torus_tube(n, r)
  phi = np.random.uniform(size=n, low=0, high=2.0*np.pi)
  res = np.c_[
    (1 + r * np.cos(theta)) * np.cos(phi), 
    (1 + r * np.cos(theta)) * np.sin(phi),
    r * np.sin(theta)
  ]
  if sd != 0:
    res += np.random.normal(size=res.shape, loc=0, scale=sd)
  return res