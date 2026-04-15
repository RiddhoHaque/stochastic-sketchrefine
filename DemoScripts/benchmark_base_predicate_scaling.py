#!/usr/bin/env python3
"""
Benchmark script: runtime, objective value, and relation size for Q3.sql as
the base predicate becomes progressively more permissive.

Two solvers are compared:
  1. RCLSolve      – default flags (no extra keyword arguments)
  2. SketchRefine  – default flags

The base predicate is varied over SELL_AFTER_LIMITS:
    WHERE SELL_AFTER <= <N> AND PRICE >= 1.0

for N in [3, 10, 20, 50, 100, 150, 200].

Each (method, N) pair is run N_ITERATIONS times and capped at TIMEOUT
seconds per run.  If any iteration times out (or produces no solution),
remaining iterations for that pair are skipped, and the same method is
also skipped for all more permissive (higher N) base predicates.

Results are written to Evaluation/benchmark_base_predicate_scaling.txt.

Run from the project root:
    python DemoScripts/benchmark_base_predicate_scaling.py
"""
import copy
import multiprocessing
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings('ignore')

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from CVaRification.RCLSolve import RCLSolve
from DbInfo.GarchPortfolioInfo import GarchPortfolioInfo
from DbInfo.GBMPortfolioInfo import GBMPortfolioInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from SketchRefine.SketchRefine import SketchRefine
from StochasticPackageQuery.Parser.Parser import Parser

SQL_PATH          = os.path.join(_PROJECT_ROOT, 'Workloads', 'DemoWorkload', 'Q3.sql')
SELL_AFTER_LIMITS = [3, 10, 20, 50, 100, 150, 200]
TIMEOUT           = 20 * 60   # seconds
N_ITERATIONS      = 5

# Each entry: (label, kind)
#   kind='rclsolve'     → _worker_rclsolve
#   kind='sketchrefine' → _worker_sketchrefine
METHODS = [
    ('RCLSolve',     'rclsolve'),
    ('SketchRefine', 'sketchrefine'),
]


# ---------------------------------------------------------------------------
# Workers (run in sub-processes to enforce timeout)
# ---------------------------------------------------------------------------

def _worker_rclsolve(query, result_queue):
    """Run RCLSolve with default flags; put (runtime, obj) in queue."""
    try:
        solver = RCLSolve(
            query=query,
            linear_relaxation=False,
            dbInfo=GBMPortfolioInfo,
            init_no_of_scenarios=Hyperparameters.INIT_NO_OF_SCENARIOS,
            no_of_validation_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
            approximation_bound=Hyperparameters.APPROXIMATION_BOUND,
            sampling_tolerance=Hyperparameters.SAMPLING_TOLERANCE,
            bisection_threshold=Hyperparameters.BISECTION_THRESHOLD,
            max_opt_scenarios=Hyperparameters.MAX_OPT_SCENARIOS_IN_PRACTICE,
        )
        package, obj_val = solver.solve()
        runtime = solver.get_metrics().get_runtime()
        result_queue.put((runtime, obj_val if package is not None else None))
    except Exception as exc:
        result_queue.put((None, None, str(exc)))


def _worker_sketchrefine(query, result_queue):
    """Run SketchRefine with default flags; put (runtime, obj) in queue."""
    try:
        solver = SketchRefine(
            query=query,
            dbInfo=GBMPortfolioInfo,
        )
        package, obj_val = solver.solve()
        runtime = solver.get_metrics().get_runtime()
        result_queue.put((runtime, obj_val if package is not None else None))
    except Exception as exc:
        result_queue.put((None, None, str(exc)))


_WORKER_FN = {
    'rclsolve':     _worker_rclsolve,
    'sketchrefine': _worker_sketchrefine,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_relation_size(relation, base_predicate):
    """Return the number of tuples satisfying the base predicate."""
    sql = f"SELECT COUNT(*) FROM {relation}"
    if base_predicate:
        sql += f" WHERE {base_predicate}"
    sql += ";"
    PgConnection.Execute(sql)
    return int(PgConnection.Fetch()[0][0])


def _run_method(query, kind):
    """Run solver in a subprocess with timeout. Returns (runtime, obj, timed_out)."""
    worker_fn = _WORKER_FN[kind]
    rq = multiprocessing.Queue()
    p  = multiprocessing.Process(
        target=worker_fn,
        args=(copy.deepcopy(query), rq)
    )
    p.start()
    p.join(timeout=TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, None, True
    result = rq.get()
    if len(result) == 3:          # error tuple (runtime, obj, exc_str)
        print(f'    Error: {result[2]}')
        return None, None, False
    return result[0], result[1], False


def _fmt_avg_std(values):
    """Return (avg_str, std_str) for a list of floats, or ('N/A', 'N/A') if empty."""
    if not values:
        return 'N/A', 'N/A'
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f'{avg:.4f}', f'{std:.4f}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(SQL_PATH, 'r') as f:
        base_query = Parser().parse(f.readlines())

    relation = base_query.get_relation()

    print(f'Query       : {SQL_PATH}')
    print(f'Relation    : {relation}')
    print(f'Timeout/run : {TIMEOUT}s')
    print(f'Iterations  : {N_ITERATIONS}\n')

    # timed_out[label] = True once a method fails/times-out at any predicate
    timed_out = {label: False for label, _ in METHODS}
    rows = []

    for sa_limit in SELL_AFTER_LIMITS:
        base_predicate = f"SELL_AFTER <= {sa_limit} AND PRICE >= 1.0"

        # Build query variant for this predicate
        q_variant = copy.deepcopy(base_query)
        q_variant.set_base_predicate(base_predicate)

        # Count tuples satisfying this base predicate
        n_tuples = _get_relation_size(relation, base_predicate)

        print(f'=== SELL_AFTER <= {sa_limit}  (relation size: {n_tuples:,}) ===')

        for label, kind in METHODS:
            if timed_out[label]:
                print(f'  [{label}] skipped (previously timed out or no solution found)')
                rows.append({
                    'sa': sa_limit, 'n_tuples': n_tuples, 'method': label,
                    'rt_avg': 'N/A', 'rt_std': 'N/A',
                    'ob_avg': 'N/A', 'ob_std': 'N/A',
                    'n_ok': 0, 'status': 'SKIPPED',
                })
                continue

            runtimes, objs = [], []
            final_status = 'OK'

            for it in range(1, N_ITERATIONS + 1):
                q = copy.deepcopy(q_variant)
                print(f'  [{label}] iter {it}/{N_ITERATIONS}...', end='', flush=True)
                t0 = time.time()
                runtime, obj, did_timeout = _run_method(q, kind)
                elapsed = time.time() - t0

                if did_timeout:
                    print(f' TIMEOUT after {elapsed:.1f}s — stopping iterations')
                    timed_out[label] = True
                    final_status = 'TIMEOUT'
                    break
                elif runtime is None:
                    print(' ERROR')
                    timed_out[label] = True
                    final_status = 'ERROR'
                    break
                elif obj is None:
                    print(f' runtime={runtime:.2f}s  obj=None — no solution, stopping')
                    timed_out[label] = True
                    final_status = 'NO_SOLUTION'
                    break
                else:
                    print(f' runtime={runtime:.2f}s  obj={obj:.4f}')
                    runtimes.append(runtime)
                    objs.append(obj)

            rt_avg, rt_std = _fmt_avg_std(runtimes)
            ob_avg, ob_std = _fmt_avg_std(objs)
            rows.append({
                'sa': sa_limit, 'n_tuples': n_tuples, 'method': label,
                'rt_avg': rt_avg, 'rt_std': rt_std,
                'ob_avg': ob_avg, 'ob_std': ob_std,
                'n_ok': len(runtimes), 'status': final_status,
            })
        print()

    # Build summary table
    col_w  = 14
    header = (f'{"SELL_AFTER":>10}  {"#Tuples":>10}  {"Method":<{col_w}}  '
              f'{"AvgRuntime(s)":>14}  {"StdRuntime(s)":>14}'
              f'  {"AvgObj":>12}  {"StdObj":>12}  {"N/Iter":>6}  {"Status":>11}')
    sep    = '-' * len(header)
    lines  = [header, sep]
    for r in rows:
        lines.append(
            f'{r["sa"]:>10}  {r["n_tuples"]:>10,}  {r["method"]:<{col_w}}  '
            f'{r["rt_avg"]:>14}  {r["rt_std"]:>14}'
            f'  {r["ob_avg"]:>12}  {r["ob_std"]:>12}  {r["n_ok"]:>6}  {r["status"]:>11}'
        )

    print('\n' + '\n'.join(lines))

    eval_dir = os.path.join(_PROJECT_ROOT, 'Evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, 'benchmark_base_predicate_scaling.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
