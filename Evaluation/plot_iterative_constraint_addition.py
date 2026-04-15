import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# Data from benchmark_iterative_constraint_addition.txt
scenarios = [10, 20, 50, 80, 100]

# Label mapping
labels = {
    "Direct R-U method":          "Direct R-U",
    "Iterative_Z":                "Incremental Z constr. insertion to ILP",
    "Iterative_Z+LP_First":       "Incremental Z constr. insertion to LP",
    "CVaROptimizerBaseline":      "Feasible CVaR Constraint Search",
    "RCLSolve (optimize_lcvar)":  "LCVaR Optimization",
}

# Methods that have successful runs (avg, std per scenario)
successful = {
    "Iterative_Z+LP_First": {
        "avg": np.array([56.7772, 59.0255, 69.0048, 71.2655, 100.5232]),
        "std": np.array([ 5.1581,  3.1466,  5.3268,  6.4933,   6.6593]),
    },
}

# Methods with no successful runs (all ERROR / SKIPPED / NO_SOLUTION / TIMEOUT)
failed_methods = [
    "Direct R-U method",
    "Iterative_Z",
    "CVaROptimizerBaseline",
    "RCLSolve (optimize_lcvar)",
]

# Colorblind-friendly palette
colors = {
    "Iterative_Z+LP_First":      "#0072B2",  # blue
    "Direct R-U method":         "#D55E00",  # vermilion
    "Iterative_Z":               "#009E73",  # green
    "CVaROptimizerBaseline":     "#CC79A7",  # pink
    "RCLSolve (optimize_lcvar)": "#E69F00",  # orange
}
markers = {
    "Iterative_Z+LP_First":      "o",
}

fig, ax = plt.subplots(figsize=(16, 8))

# Plot successful methods
for key, data in successful.items():
    avg = data["avg"]
    std = data["std"]
    color = colors[key]
    marker = markers[key]
    label = labels[key]
    ax.plot(scenarios, avg, color=color, marker=marker, linestyle="-",
            linewidth=5, label=label)
    ax.fill_between(scenarios, avg - std, avg + std,
                    color=color, alpha=0.2)

# Build legend entries for failed methods (red X)
failed_handles = []
for key in failed_methods:
    handle = mlines.Line2D(
        [], [],
        color="red",
        marker="x",
        linestyle="None",
        markersize=12,
        markeredgewidth=2.5,
        label=labels[key],
    )
    failed_handles.append(handle)

# Collect existing handles from ax
existing_handles, existing_labels = ax.get_legend_handles_labels()
all_handles = existing_handles + failed_handles
all_labels  = [h.get_label() for h in all_handles]

ax.set_xlabel("# of Scenarios", fontsize=24)
ax.set_ylabel("Runtime (s)", fontsize=24)
ax.set_xticks(scenarios)
ax.tick_params(axis="both", labelsize=20)
ax.legend(handles=all_handles, labels=all_labels, fontsize=26, loc="lower right")
ax.set_ylim(bottom=0)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = "Evaluation/benchmark_iterative_constraint_addition.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()
