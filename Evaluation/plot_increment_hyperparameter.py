import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark_increment_hyperparameter.txt
increments = [1, 5, 10, 15, 20, 50, 100]

primal_avg = [98.2100, 81.4701, 80.2458, 77.6604, 80.0443, 76.3021, 74.5884]
primal_std = [ 3.8300,  4.1842,  2.0295,  2.0219,  6.8836,  2.4303,  4.4212]

dual_avg   = [97.6937, 76.1919, 79.4242, 80.7821, 82.2341, 76.9446, 78.9984]
dual_std   = [ 1.5524,  2.0406,  1.8256,  2.8632,  3.5143,  3.0683,  2.4999]

pdhg_avg   = [105.1771, 88.0504, 83.3680, 82.3171, 84.2851, 81.1051, 80.1930]
pdhg_std   = [  2.1840,  1.7052,  2.5784,  2.2225,  1.7950,  2.0910,  1.1912]

# Colorblind-friendly palette (Wong 2011)
color_primal = "#CC0000"  # red
color_dual   = "#00AA00"  # green
color_pdhg   = "#0000CC"  # blue

primal_avg = np.array(primal_avg)
primal_std = np.array(primal_std)
dual_avg   = np.array(dual_avg)
dual_std   = np.array(dual_std)
pdhg_avg   = np.array(pdhg_avg)
pdhg_std   = np.array(pdhg_std)

# 95th percentile estimate via CLT: avg + z(0.95) * std
Z95 = 1.645
all_p95 = {
    "Primal Simplex": primal_avg + Z95 * primal_std,
    "Dual Simplex":   dual_avg   + Z95 * dual_std,
    "PDHG (GPU)":     pdhg_avg   + Z95 * pdhg_std,
}
best_p95_val = np.inf
best_p95_method = ""
best_p95_inc = 0
for method, p95_arr in all_p95.items():
    idx = np.argmin(p95_arr)
    if p95_arr[idx] < best_p95_val:
        best_p95_val = p95_arr[idx]
        best_p95_method = method
        best_p95_inc = increments[idx]

fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(increments, pdhg_avg, color=color_pdhg, marker="^", linestyle=":", linewidth=5, label="PDHG (GPU)")
ax.fill_between(increments, pdhg_avg - pdhg_std, pdhg_avg + pdhg_std,
                color=color_pdhg, alpha=0.2)

ax.plot(increments, dual_avg, color=color_dual, marker="s", linestyle="--", linewidth=5, label="Dual Simplex")
ax.fill_between(increments, dual_avg - dual_std, dual_avg + dual_std,
                color=color_dual, alpha=0.2)

ax.plot(increments, primal_avg, color=color_primal, marker="o", linestyle="-", linewidth=5, label="Primal Simplex")
ax.fill_between(increments, primal_avg - primal_std, primal_avg + primal_std,
                color=color_primal, alpha=0.2)

ax.set_title(
    f"Best Avg. Runtime: Primal Simplex with all constraints added at once (74.59s)\n"
    f"Best Tail Latency (P$_{{95}}$): {best_p95_method} at {best_p95_inc} constraints/iteration ({best_p95_val:.2f}s)",
    fontsize=24, pad=20
)
ax.set_xlabel("# of constraints added per iteration", fontsize=24)
ax.set_ylabel("Runtime (secs)", fontsize=24)
ax.set_xticks(increments)
ax.tick_params(axis="both", labelsize=20)
ax.legend(fontsize=26)
ax.set_ylim(60, 110)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = "benchmark_increment_hyperparameter.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()
