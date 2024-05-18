# Computes bias in the median of the sample quantiles by each of the
# "canonical" continuous definitions.

import numpy as np
from scipy.stats import uniform, norm, pareto
from scipy.stats.mstats import mquantiles
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

N_trials  = 100000
N_samples = [100]
levels    = np.linspace(0.01, 0.99, num = 5000)

use_method = [4, 5, 6, 7, 8, 9]
#use_distribution = ["Uniform", "Normal", "Pareto (b=2)"]
use_distribution = ["Uniform"]
suffix = ""
# A copy-paste artefact from the mean-bias code:
highlight_order = [0, 1, 3, 4, 5, 2]

distributions = [
  {
    "name": "Uniform",
    "dist": uniform,
    "exact_line_bounds": [0., 1.],
  },
  {
    "name": "Normal",
    "dist": norm,
    "exact_line_bounds": [0.01, 0.99],
  },
  {
    "name": "Pareto (b=2)",
    "dist": pareto(b = 2),
    "exact_line_bounds": [0.01, 0.99],
  }
]

methods = [
  {"alpha": 0.0, "beta": 1.0, "R": 4},
  {"alpha": 0.5, "beta": 0.5, "R": 5},
  {"alpha": 0.0, "beta": 0.0, "R": 6},
  {"alpha": 1.0, "beta": 1.0, "R": 7},
  {"alpha": 1/3, "beta": 1/3, "R": 8},
  {"alpha": 3/8, "beta": 3/8, "R": 9}
]

methods = [m for m in methods if m["R"] in use_method]
distributions = [d for d in distributions if d["name"] in use_distribution]

true_values = {}
print("Computing true quantile values")
for distribution in distributions:
  true_values[distribution["name"]] = distribution["dist"].ppf(levels)

median_bias = {}

for i_method, method in enumerate(methods):
  median_bias[i_method] = {}
  for distribution in distributions:
    median_bias[i_method][distribution["name"]] = {}
    for N in N_samples:
      median_bias[i_method][distribution["name"]][N] = np.zeros_like(levels)

for i_trial in range(N_trials):
  if i_trial % 100 == 0:
    print(f"Starting trial {i_trial}")
  
  for N in N_samples:
    x = np.random.random(N)

    for distribution in distributions:
      u = distribution["dist"].ppf(x)
      for i_method, method in enumerate(methods):
        q = mquantiles(u, levels, alphap = method["alpha"], betap = method["beta"])
        median_bias[i_method][distribution["name"]][N] += (q > true_values[distribution["name"]])

for N in N_samples:
  for distribution in distributions:
    for i_method, _ in enumerate(methods):
      median_bias[i_method][distribution["name"]][N] /= N_trials


mean_order_statistics = {}
fade_color_weight = 0.85

for N in N_samples:
  plot_k = np.array([k for k in range(1, N + 1)])
  mean_order_statistics[N] = {}

  for distribution in distributions:
    name = distribution["name"]
    dist = distribution["dist"]
    
    plt.close("all")
    fig, ax = plt.subplots()

    print("Plotting graphs")
    ax.plot(distribution["exact_line_bounds"], [0.5, 0.5], color = "black", label = "Median-unbiased", zorder = -1)

    lines = {}
    colors = {}
    faded_colors = {}

    for i_method, method in enumerate(methods):
      label = f"R-{method['R']}"

      lines[label], = ax.plot(levels, median_bias[i_method][distribution["name"]][N], label = label, zorder = 1)
      colors[label] = mcolors.to_rgb(lines[label].get_color())
      faded_colors[label] = tuple(fade_color_weight*(1 - c) + c for c in colors[label])

    ax.set_ylabel("Fraction overestimated")
    ax.set_xlabel("Quantile level")
    ax.legend()

    fig.suptitle(f"Median-biasedness, {name}, N = {N}")
    fig.tight_layout()
    fig.savefig(f"bias_median_{name}_N_{N}{suffix}.png")

    # Now highlight each curve in turn

    for method in methods:
      label = f"R-{method['R']}"
      lines[label].set_color(faded_colors[label])
      lines[label].set_zorder(0)
    
    for method in [methods[i] for i in highlight_order]:
      label = f"R-{method['R']}"
      lines[label].set_color(colors[label])
      lines[label].set_zorder(2)
      lines[label].set_linewidth(2.)
      ax.legend()
      fig.savefig(f"bias_median_{name}_N_{N}_{label}{suffix}.png")
      lines[label].set_color(faded_colors[label])
      lines[label].set_zorder(1)
      lines[label].set_linewidth(1.)


