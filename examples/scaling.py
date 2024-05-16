import numpy as np 
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import row
output_notebook()
from math import comb 

import json
# prof_timings = json.load(open("/Users/mpiekenbrock/laplacian_kernel/timings_gpu.json", "r"))
prof_timings = json.load(open("/Users/mpiekenbrock/laplacian_kernel/data/timings_a100.json", "r"))
# [int(2**(k)) for k in np.arange(3, 11, 0.5)]
K = np.array([3,4,5,6,7])
N = np.array([22, 32, 45, 64, 90, 128, 181, 256, 362, 512, 724])

from scipy.interpolate import interp1d
results = []
for k in K: 
  for i, n in enumerate(N): 
    key = f'({n}, {k})'
    if key not in prof_timings: 
      print(key)
      continue
      # raise ValueError("here")
      # xp, xpp = N[i-1], N[i-2]
      # yp = np.mean(prof_timings[f'({xp}, {k})'])
      # ypp = np.mean(prof_timings[f'({xpp}, {k})'])
      # f = interp1d(np.log2([xpp, xp]), np.log10([ypp, yp]), kind='linear', fill_value='extrapolate')
      # # ye = ypp + (n - xpp)/(xp - xpp) * (yp - ypp)
      # ye = f(n)
      # prof_timings[f'({n}, {k})'] = np.array([ye])
      # results.append((n,k,ye,True))
    else: 
      results.append((n,k,np.mean(prof_timings[key]), False))
results = np.array(results, dtype=[('n', 'i4'), ('k', 'i4'), ('time', 'f4'), ('extra', 'bool')])  

from scipy.interpolate import interp1d
results_dict = {}
for k in K:
  k_results = results[results['k'] == k]
  f = interp1d(np.log2(k_results['n']), np.log10(k_results['time']),  kind='linear', fill_value='extrapolate')
  extrapolated = np.zeros(len(N), dtype=bool)
  extrapolated[len(k_results['n']):] = True
  results_dict[k] = np.array(list(zip(N, np.repeat(k, len(N)), 10**(f(np.log2(N))), extrapolated)), dtype=[('n', 'i4'), ('k', 'i4'), ('time', 'f4'), ('extra', 'bool')])

# 10**(f(np.log2(N)))

# p = figure(width=250, height=250, y_axis_type="log", x_axis_type="log")
# p.scatter(N, 10**(f(np.log2(N))))
# show(p)


# from scipy.interpolate import interp1d
# k_results = results[results[:,1] == k]
# # f = interp1d(k_results[:3,0], k_results[:3,2], kind='linear', fill_value='extrapolate')
# f = interp1d(np.log2(k_results[:3,0]), np.log2(k_results[:3,2]), kind='linear', fill_value='extrapolate')
# 2**(f(np.log2(k_results[3:,0])))


from bokeh.models import LogTicker, LogTickFormatter

K_colors = ["orange", "red", "#8B8000", "blue", "green", "purple"]

p = figure(
  width=400, height=300, y_axis_type="log", x_axis_type="log", 
  title="Laplacian Kernel time (A100)", 
  x_axis_label="Number of points", y_axis_label="Matvec time (seconds)"
)
p.title.align = 'center'
for j, k in enumerate(K): 
  # k_results = results[results[:,1] == k]
  k_results = results_dict[k]
  is_extrapolated = k_results['extra']
  if any(is_extrapolated):
    extrapolated_ind = np.flatnonzero(is_extrapolated)
    extrapolated_ind = np.append([extrapolated_ind[0]-1], extrapolated_ind)
    p.line(k_results[~is_extrapolated]['n'], k_results[~is_extrapolated]['time'], color=K_colors[j], line_dash='solid', legend_label=f"{k}")
    p.line(k_results[extrapolated_ind]['n'], k_results[extrapolated_ind]['time'], color=K_colors[j], line_dash='dotted')
    p.scatter(k_results['n'], k_results['time'], color=K_colors[j], marker=list(np.where(is_extrapolated, "square", "circle")))
  else:
    p.line(k_results['n'], k_results['time'], color=K_colors[j], line_dash='solid', legend_label=f"{k}")
    p.scatter(k_results['n'], k_results['time'], color=K_colors[j], marker="circle")
p.legend.location = "top_left"
p.legend.spacing = 0
p.legend.title = "d + 1"
p.legend.padding = 5
p.legend.margin = 5
p.toolbar_location = None
p.xaxis[0].ticker = LogTicker(base=2.0)
p.xaxis[0].formatter = LogTickFormatter()
show(p)

from math import comb
from scipy.special import comb
q = figure(
  width=400, height=300, y_axis_type="log", x_axis_type="log", 
  title = "Simplicial complex size",
  x_axis_label="Number of points", y_axis_label="Number of d-simplices"
)
q.title.align = 'center'

from bokeh.models import BoxAnnotation, Text
from scipy.optimize import minimize, golden
from bokeh.models import Span
obj = lambda n: np.abs(((n * 4) // 1024**3) - 40.0)**2
s_UB = int(golden(obj, brack=(1e10, 1e14)))
box = BoxAnnotation(top=s_UB, fill_alpha=0.10, fill_color='red', line_color='red', line_dash="dashed")
q.add_layout(box)

# glyph = Text(x=128, y=10**4, text="40GB ", angle=0.3, text_color="#96deb3")
# q.add_glyph(glyph)


# q.toolbar_location = None
for j, k in enumerate(K): 
  k_results = results_dict[k]
  is_extrapolated = k_results['extra']
  if any(is_extrapolated):
    extrapolated_ind = np.flatnonzero(is_extrapolated)
    extrapolated_ind = np.append([extrapolated_ind[0]-1], extrapolated_ind)
    IN = k_results[~is_extrapolated]['n']
    EN = k_results[extrapolated_ind]['n']
    IK, EK = comb(IN, k), comb(EN, k)
    q.line(IN, IK, color=K_colors[j], line_dash='solid', legend_label=f"{k}")
    q.line(EN, EK, color=K_colors[j], line_dash='dotted')
    q.scatter(k_results['n'], comb(k_results['n'], k), color=K_colors[j], marker=list(np.where(is_extrapolated, "square", "circle")))
  else:
    q.line(k_results['n'], comb(k_results['n'], k), color=K_colors[j], line_dash='solid', legend_label=f"{k}")
    q.scatter(k_results['n'], comb(k_results['n'], k), color=K_colors[j], marker="circle")


# sp = Span(location=s_UB, dimension='width', line_dash="dashed", line_color="gray", line_width=2.5)
# q.add_layout(sp)
# q.rect(width=float(max(N)), height=s_UB, x=max(N)/2.0, y=s_UB/2.0, fill_alpha=0.50, fill_color='red', line_dash="dashed", color='red')
q.legend.location = "top_left"
q.legend.spacing = 0
q.legend.title = "d + 1"
q.legend.padding = 5
q.legend.margin = 5
q.xaxis[0].ticker = LogTicker(base=2.0)
q.xaxis[0].formatter = LogTickFormatter()

show(q)
show(row(p, q))



# 
# UB = {}
# for k in K: 
#   obj = lambda n: np.abs(((comb(n,k-1) * 8) // 1024**3) - 40.0)**2
#   n_ub = int(golden(obj, brack=(1, 4096*2)))
#   UB[k] = n_ub


((comb(512,3)*8) / 1024**3)
2**11

from ripser import ripser
X = np.random.uniform(size=(128, 2))
ripser(X, maxdim=5)

65535

2**16