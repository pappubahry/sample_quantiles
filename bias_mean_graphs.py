# Computes bias in the mean of the sample quantiles by each of the
# "canonical" continuous definitions.
#
# The uniform case could be treated exactly, rather than using numerical
# intergration.
#
# R = 10 isn't a real definition available in R.

import numpy as np
from scipy.stats import uniform, norm, pareto
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

N_samples = [100]
levels    = np.linspace(0.01, 0.99, num = 5000)

# Comparison of R-5 with the definitions targeting the normal distribution:
# use_method = [5, 9, 10]
# use_distribution = ["Normal"]
# suffix = "_5v9"
# highlight_order = [0, 1, 2]

use_method = [4, 5, 6, 7, 8, 9]
use_distribution = ["Uniform", "Normal", "Pareto (b=2)"]
suffix = ""
# An ugly hack to make the R-6 highlight plot last,
# because otherwise its colour gets mixed up with the
# black 'Exact' line:
highlight_order = [0, 1, 3, 4, 5, 2]

distributions = [
  {
    "name": "Uniform",
    "dist": uniform,
    "bounds": [0., 1.],
    "exact_line_bounds": [0., 1.],
    "axis_ticks":      [ 0.0,   0.2,   0.4,   0.6,   0.8,   1.0],
    "axis_ticklabels": ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
    "error_metric": lambda actual, theory: actual - theory,
    "title": "Bias"
  },
  {
    "name": "Normal",
    "dist": norm,
    "bounds": [-9., 9.],
    "exact_line_bounds": [0.01, 0.99],
    "axis_ticks":      [ 0.01,   0.05,   0.25,   0.50,   0.75,   0.95,   0.99],
    "axis_ticklabels": ["0.01", "0.05", "0.25", "0.50", "0.75", "0.95", "0.99"],
    "error_metric": lambda actual, theory: actual - theory,
    "title": "Bias"
  },
  {
    "name": "Pareto (b=2)",
    "dist": pareto(b = 2),
    "bounds": [ 1., np.inf],
    "exact_line_bounds": [0.01, 0.99],
    "axis_ticks":      [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  0.8,   0.9,   0.95,   0.98,   0.99],
    "axis_ticklabels": ["0.1", "",  "",  "",  "",  "",  "", "0.8", "0.9", "0.95", "0.98", "0.99"],
    "error_metric": lambda actual, theory: 100 * (actual / theory - 1),
    "title": "Percentage bias"
  }
]

methods = [
  {"alpha": 0.0, "beta": 1.0, "R": 4},
  {"alpha": 0.5, "beta": 0.5, "R": 5},
  {"alpha": 0.0, "beta": 0.0, "R": 6},
  {"alpha": 1.0, "beta": 1.0, "R": 7},
  {"alpha": 1/3, "beta": 1/3, "R": 8},
  {"alpha": 3/8, "beta": 3/8, "R": 9},
  {"alpha": 0.4, "beta": 0.4, "R": 10}
]

methods = [m for m in methods if m["R"] in use_method]
distributions = [d for d in distributions if d["name"] in use_distribution]

def log_factorial(n):
  return np.sum(np.log(np.arange(1, n + 1)))

def wrapper(N, k, dist):
  # Returns a function whose integral is the expectation of the k-th order statistic.
  log_factorials = log_factorial(N) - log_factorial(k - 1) - log_factorial(N - k)

  def integrand(t):
    log1 = dist.logcdf(t)
    log2 = dist.logsf(t)
    log3 = dist.logpdf(t)
    return t * np.exp(log_factorials + (k - 1)*log1 + (N - k)*log2 + log3)
 
  return integrand

true_values = {}
print("Computing true quantile values")
for distribution in distributions:
  true_values[distribution["name"]] = distribution["dist"].ppf(levels)

mean_order_statistics = {}
fade_color_weight = 0.85

standard_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for N in N_samples:
  plot_k = np.array([k for k in range(1, N + 1)])
  mean_order_statistics[N] = {}

  for distribution in distributions:
    name = distribution["name"]
    title = distribution["title"]
    print(f"Computing order statistic expectations for {name}")
    mean_order_statistics[N][name] = np.zeros(N)

    dist = distribution["dist"]
    x0 = distribution["bounds"][0]
    x1 = distribution["bounds"][1]
    for k in range(N):
      mean_order_statistics[N][name][k], _ = quad(wrapper(N, k + 1, dist), x0, x1)
    
    plt.close("all")
    fig, ax = plt.subplots()

    print("Plotting graphs")
    ax.plot(dist.ppf(distribution["exact_line_bounds"]), [0., 0.], color = "black", label = "Exact", zorder = -1)

    lines = {}
    colors = {}
    faded_colors = {}

    for method in methods:
      alpha = method["alpha"]
      beta = method["beta"]
      label = f"R-{method['R']}"
      i_color = method["R"] - 4
      plot_p = (plot_k - alpha) / (N + 1 - alpha - beta)
      mean_q = np.interp(levels, plot_p, mean_order_statistics[N][name])

      errors = distribution["error_metric"](mean_q, true_values[name])
      lines[label], = ax.plot(dist.ppf(levels), errors, label = label, zorder = 1, color = standard_colors[i_color])
      colors[label] = mcolors.to_rgb(lines[label].get_color())
      faded_colors[label] = tuple(fade_color_weight*(1 - c) + c for c in colors[label])

      ax.set_xticks(dist.ppf(distribution["axis_ticks"]))
      ax.set_xticklabels(distribution["axis_ticklabels"])

    ax.set_ylabel(title)
    ax.set_xlabel("Quantile level")
    ax.legend()

    fig.suptitle(f"{title}, {name}, N = {N}")
    fig.tight_layout()
    fig.savefig(f"bias_mean_{name}_N_{N}{suffix}.png")

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
      fig.savefig(f"bias_mean_{name}_N_{N}_{label}{suffix}.png")
      lines[label].set_color(faded_colors[label])
      lines[label].set_zorder(1)
      lines[label].set_linewidth(1.)



