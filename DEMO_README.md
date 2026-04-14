# Demo Setup and Benchmarking Guide

This guide explains how to populate the portfolio tables, run the benchmarking
script, generate comparison plots, and run the demo query workload.

---

## Prerequisites

### Python packages

```bash
pip install psycopg2-binary yfinance arch pandas pandas_market_calendars matplotlib numpy torch gurobipy psutil
```

> **Note:** `torch` is an optional dependency used for GPU-accelerated matrix
> multiplication when a CUDA-capable GPU is available (for iterative Z-constraint
> checking and the LP-first solve mode). If `torch` is not installed or no GPU is
> present, the code falls back to vectorised NumPy automatically.

### PostgreSQL

A running PostgreSQL instance is required. Edit `Data/database.ini` with your
connection details:

```ini
[postgresql]
dbname   = your_database
user     = your_user
password = your_password
host     = your_host
port     = your_port
```

### Ticker data

The fetchers discover tickers from local CSV directories:

```text
Data/Portfolio/stock_market_data/nasdaq/csv/<TICKER>.csv
Data/Portfolio/stock_market_data/nyse/csv/<TICKER>.csv
```

Each file just needs to exist; the filename is used as the ticker symbol.
Actual price data is downloaded from Yahoo Finance at runtime.

---

## Step 1 - Populate the database tables

Run the setup script from the project root.

On Linux / macOS:

```bash
bash Data/setup_demo_data.sh
```

On Windows:

```bat
Data\setup_demo_data.bat
```

This script performs five steps in order:

| Step | What it does |
|------|--------------|
| 1 | Creates `GARCH_Portfolio_YYYY` and `GARCH_Residuals_YYYY` tables (2015-2025) |
| 2 | Creates `GBM_Portfolio_YYYY` tables (2015-2025) |
| 3 | Downloads historical prices from Yahoo Finance, fits GARCH(1,1) per ticker/year, and populates the GARCH tables |
| 4 | Downloads historical prices, fits GBM per ticker/year, and populates the GBM tables |
| 5 | Reconciles both table families so they contain the same `(ticker, sell_after)` pairs |

Actual closing prices fetched during steps 3-4 are cached in the `ActualClose`
and `ActualCloseDownloaded` PostgreSQL tables. Re-running the script skips any
data already cached.

> **Note:** Steps 3 and 4 download several years of daily price data for every
> discovered ticker. The first run can take a while; subsequent runs use the
> cache and are much faster.

---

## Step 2 - Partition the portfolio relations

From the project root, run **only** the following command (passing the year `2025`):

```bash
python DemoScripts/demo_partitioning.py 2025
```

This partitions the `GBM_Portfolio_2025` and `GARCH_Portfolio_2025` tables if
they exceed `Hyperparameters.SIZE_THRESHOLD` rows, using pre-configured gain and
price diameter thresholds. Tables below the threshold are left untouched.

Console output looks like:

```text
SIZE_THRESHOLD        = <N>
PARTITION_COUNT_LIMIT = <0.8 * N>

GBM relations above SIZE_THRESHOLD:   ['GBM_Portfolio_2025', ...]
GARCH relations above SIZE_THRESHOLD: ['GARCH_Portfolio_2025', ...]
```

Followed by per-relation partitioning metrics.

> **Note:** If you want to search for optimal diameter thresholds before
> partitioning, run `gbm_fhs_diameter_threshold_search.py` first and update
> `Hyperparameters` with the results.

---

## Step 3 - Run the benchmark

From the project root:

```bash
python DemoScripts/benchmark_gain_methods.py
```

The script:

- Generates `100` gain scenarios using both GBM and GARCH-FHS for the
  year-pairs `2020-21` through `2025-26`.
- Computes the per-ticker price-normalized gain error
  `abs(predicted_gain - actual_gain) / price` at sell-after days
  `10, 20, 30, ..., 100`.
- Aggregates price-normalized RMSE values for percentile cutoffs
  `50`, `75`, `90`, `95`, and `100`.
- Caches any missing actual closing prices in PostgreSQL.
- Prints a runtime summary to the console.
- Writes all results to `Evaluation/benchmark_results.json`.

Console output looks like:

```text
=== Year 2020-2021 ===
  GBM rows: 42,168  |  GARCH param rows: 38,520
  GBM       generated in 1.43s
  GARCH-FHS generated in 4.87s
  SA= 10: GBM=167 tickers, GARCH-FHS=153 tickers, price-normalized RMSE(100th pct) GBM=0.196300, GARCH-FHS=0.219000
  SA= 20: GBM=167 tickers, GARCH-FHS=153 tickers, price-normalized RMSE(100th pct) GBM=0.209500, GARCH-FHS=2.942200
  ...
  SA=100: GBM=167 tickers, GARCH-FHS=153 tickers, price-normalized RMSE(100th pct) GBM=0.475500, GARCH-FHS=0.307400
...

=======================================================
Runtime Summary  (100 scenarios per call)
=======================================================
  GBM        mean=1.52s  median=1.48s  min=1.38s  max=1.71s
  GARCH-FHS  mean=5.14s  median=5.02s  min=4.87s  max=5.52s
=======================================================

Results written to Evaluation/benchmark_results.json
```

---

## Step 4 - Generate plots

From the project root:

```bash
python DemoScripts/plot_benchmark_results.py
```

This reads `Evaluation/benchmark_results.json` and writes one year-specific
figure per benchmark year plus a runtime comparison chart.

| File | Description |
|------|-------------|
| `Evaluation/price_normalized_rmse_percentile_lines_2020.png` ... `Evaluation/price_normalized_rmse_percentile_lines_2025.png` | One figure per year-pair. Each figure contains line charts for `P50`, `P75`, `P90`, and `P95` price-normalized RMSE, with sell-after day on the X-axis and price-normalized RMSE on a linear Y-axis capped at `0.2`. Each subplot compares GBM (blue) vs GARCH-FHS (orange). |
| `Evaluation/runtime_comparison.png` | Grouped bar chart comparing scenario generation runtimes by year-pair, with mean reference lines. |
| `Evaluation/summary_statistics.txt` | Text summary table of price-normalized RMSE values for `P50`, `P75`, `P90`, and `P95` across all sell-after days and years. |

---

## Step 5 - Run the solver benchmarks

Six benchmark scripts measure different aspects of the RCLSolve solver on Q2.sql.
Run each from the project root:

### 5a — Iterative Z constraint addition vs scenario count

```bash
python DemoScripts/benchmark_iterative_constraint_addition.py
```

Varies the number of optimisation scenarios over `[10, 20, 50, 80, 100]` and
compares three solver configurations:

| Configuration | Flags |
|---|---|
| `Default_RCLSolve` | *(none)* |
| `Iterative_Z` | `iterative_constraint_addition=True` |
| `Iterative_Z+LP_First` | `iterative_constraint_addition=True, solve_lp_first=True` |

A method is skipped for higher scenario counts once it times out (20-minute cap per run).
Results are saved to `Evaluation/benchmark_iterative_constraint_addition.txt`.

### 5b — Increment hyperparameter sweep

```bash
python DemoScripts/benchmark_increment_hyperparameter.py
```

Fixes the scenario count at 50 and varies `increment_in_number_of_constraints`
over `[1, 5, 10, 15, 20]` with `iterative_constraint_addition=True` and
`solve_lp_first=True`.
Results are saved to `Evaluation/benchmark_increment_hyperparameter.txt`.

### 5c — Random vs heuristic initial Z constraints

```bash
python DemoScripts/benchmark_random_initial_constraints.py
```

Compares two initialisation strategies for the iterative Z constraint addition
under `solve_lp_first=True`:

| Configuration | Flag |
|---|---|
| `LP_First / heuristic_init` | `set_initial_constraints_randomly=False` |
| `LP_First / random_init` | `set_initial_constraints_randomly=True` |

Results are saved to `Evaluation/benchmark_random_initial_constraints.txt`.

### 5d — GPU vs CPU

```bash
python DemoScripts/benchmark_use_gpu.py
```

Compares `use_gpu=True` (PDHG LP solver + GPU matrix multiply) against
`use_gpu=False` (CPU only), both with `solve_lp_first=True`.
Results are saved to `Evaluation/benchmark_use_gpu.txt`.

### 5e — Dual simplex vs primal simplex

```bash
python DemoScripts/benchmark_solve_dual_lp.py
```

Runs with `use_gpu=False` and compares `solve_dual_lp=True` (Gurobi dual
simplex, Method=1) against `solve_dual_lp=False` (primal simplex, Method=0),
both with `solve_lp_first=True`.
Results are saved to `Evaluation/benchmark_solve_dual_lp.txt`.

---

## Step 6 - Run the demo query workload

From the project root:

```bash
python DemoScripts/QueryRunner.py
```

The script:

- Loads the demo SQL workload from `Workloads/DemoWorkload/Q1.sql`.
- Runs `SketchRefine` on `GBM_Portfolio_2021` through `GBM_Portfolio_2025`.
- Prints the selected package tuples with all fetched attributes.
- Runs `BackTest` on each returned package and prints:
  actual gain
  total investment
  benchmark profit comparisons for DOW, S&P500, NASDAQ, and Russell2000
- Prints a summary table across all processed relations at the end.

Console output starts like:

```text
Query file: ...\Workloads\DemoWorkload\Q1.sql

========================================================================
  GBM_Portfolio
========================================================================

--- GBM_Portfolio_2021 ---

  [SketchRefine / Q1]
    Objective value : ...
    Runtime (s)     : ...
```

If a relation does not exist, the script skips it. If a solver run exceeds the
configured timeout, the worker process is terminated and the summary marks that
relation as timed out.

---

## File overview

```text
Data/
  setup_demo_data.sh               # One-shot setup script (Linux/macOS)
  setup_demo_data.bat              # One-shot setup script (Windows)
  garch_init.sql                   # GARCH table schemas
  gbm_init.sql                     # GBM table schemas
  garch_table_builder.py           # Executes garch_init.sql via Python
  gbm_table_builder.py             # Executes gbm_init.sql via Python
  reconcile_portfolio_tables.py    # Reconciles GBM/GARCH (ticker, sell_after) pairs
  Portfolio/
    yfinance_garch_fetcher.py      # Fits GARCH(1,1) and populates GARCH tables
    yfinance_gbm_fetcher.py        # Fits GBM and populates GBM tables

DemoScripts/
  demo_partitioning.py             # Partitions qualifying portfolio relations
  benchmark_gain_methods.py        # Runs benchmark, writes JSON
  plot_benchmark_results.py        # Reads JSON, writes PNG plots
  QueryRunner.py                   # Runs the demo SQL workload with SketchRefine
  benchmark_iterative_constraint_addition.py  # Script 5a: runtime/objective vs optimisation scenario count
  benchmark_increment_hyperparameter.py       # Script 5b: runtime/objective vs increment_in_number_of_constraints
  benchmark_random_initial_constraints.py     # Script 5c: heuristic vs random initial Z constraints
  benchmark_use_gpu.py                        # Script 5d: GPU vs CPU LP solve
  benchmark_solve_dual_lp.py                 # Script 5e: dual simplex vs primal simplex LP solve

Workloads/DemoWorkload/
  Q1.sql                           # Original demo query (GBM_Portfolio_2025)
  Q2.sql                           # CVaR-objective query on Garch_Portfolio_2023
gbm_fhs_diameter_threshold_search.py  # Searches for optimal diameter thresholds
Evaluation/
  benchmark_results.json           # Output of benchmark (auto-created)
  price_normalized_rmse_percentile_lines_2020.png  # Output of plot script (auto-created)
  price_normalized_rmse_percentile_lines_2021.png  # Output of plot script (auto-created)
  price_normalized_rmse_percentile_lines_2022.png  # Output of plot script (auto-created)
  price_normalized_rmse_percentile_lines_2023.png  # Output of plot script (auto-created)
  price_normalized_rmse_percentile_lines_2024.png  # Output of plot script (auto-created)
  price_normalized_rmse_percentile_lines_2025.png  # Output of plot script (auto-created)
  runtime_comparison.png           # Output of plot script (auto-created)
  summary_statistics.txt           # Output of plot script (auto-created)
```
