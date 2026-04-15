#!/usr/bin/env python3
"""
Benchmark script 5: runtime and objective value for Q3.sql as the number of
optimisation scenarios increases.

Five solver configurations are compared:
  1. Direct R-U method                        (default RCLSolve, no extra flags)
  2. Iterative_Z                              (iterative_constraint_addition=True)
  3. Iterative_Z+LP_First                     (iterative_constraint_addition=True,
                                               solve_lp_first=True)
  4. CVaROptimizerBaseline                    (binary-search baseline solver)
  5. RCLSolve (optimize_lcvar=True)           (optimize_lcvar=True)

The number of optimisation scenarios is varied over SCENARIO_COUNTS.
Each (method, scenario-count) pair is run N_ITERATIONS times and capped at
TIMEOUT seconds per run.  If any iteration times out (or no solution is found),
remaining iterations for that pair are skipped and the method is also skipped
at all higher scenario counts.  Average and standard deviation of runtime and
objective value are reported.

RCLSolve-based methods are configured so the solver cannot increase the
scenario count beyond the value under test
(max_opt_scenarios = init_no_of_scenarios = N).

Run from the project root:
    python DemoScripts/benchmark_iterative_constraint_addition.py
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
from CVaROptimizer.CVaROptimizerBaseline import CVaROptimizerBaseline
from DbInfo.GarchPortfolioInfo import GarchPortfolioInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from StochasticPackageQuery.Parser.Parser import Parser

SQL_PATH        = os.path.join(_PROJECT_ROOT, 'Workloads', 'DemoWorkload', 'Q3.sql')
SCENARIO_COUNTS = [10, 20, 50, 80, 100]
TIMEOUT         = 20 * 60   # seconds
N_ITERATIONS    = 5

# Each entry: (label, kind, extra_kwargs)
#   kind='rclsolve'  → use _worker_rclsolve (respects n_scenarios)
#   kind='baseline'  → use _worker_baseline (CVaROptimizerBaseline)
METHODS = [
    ('Direct R-U method',           'rclsolve',  {}),
    ('Iterative_Z',                 'rclsolve',  {'iterative_constraint_addition': True}),
    ('Iterative_Z+LP_First',        'rclsolve',  {'iterative_constraint_addition': True,
                                                   'solve_lp_first': True}),
    ('CVaROptimizerBaseline',       'baseline',  {}),
    ('RCLSolve (optimize_lcvar)',   'rclsolve',  {'optimize_lcvar': True}),
]


# ---------------------------------------------------------------------------
# Worker functions (run in sub-processes)
# ---------------------------------------------------------------------------

def _worker_rclsolve(query, n_scenarios, extra_kwargs, result_queue):
    """Worker: instantiate RCLSolve and run it; put (runtime, obj) in queue."""
    try:
        solver = RCLSolve(
            query=query,
            linear_relaxation=False,
            dbInfo=GarchPortfolioInfo,
            init_no_of_scenarios=n_scenarios,
            no_of_validation_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
            approximation_bound=Hyperparameters.APPROXIMATION_BOUND,
            sampling_tolerance=Hyperparameters.SAMPLING_TOLERANCE,
            bisection_threshold=Hyperparameters.BISECTION_THRESHOLD,
            max_opt_scenarios=n_scenarios,
            **extra_kwargs
        )
        package, obj_val = solver.solve()
        runtime = solver.get_metrics().get_runtime()
        result_queue.put((runtime, obj_val if package is not None else None))
    except Exception as exc:
        result_queue.put((None, None, str(exc)))


def _worker_baseline(query, n_scenarios, _extra_kwargs, result_queue):
    """Worker: instantiate CVaROptimizerBaseline and run it; put (runtime, obj) in queue."""
    try:
        t0 = time.time()
        solver = CVaROptimizerBaseline(
            query=query,
            dbInfo=GarchPortfolioInfo,
            num_val_scenarios=n_scenarios,
        )
        package, obj_val = solver.solve()
        runtime = time.time() - t0
        result_queue.put((runtime, obj_val if package is not None else None))
    except Exception as exc:
        result_queue.put((None, None, str(exc)))


_WORKER_FN = {
    'rclsolve': _worker_rclsolve,
    'baseline': _worker_baseline,
}


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def _run_method(query, n_scenarios, kind, extra_kwargs):
    """Run a solver in a subprocess with timeout. Returns (runtime, obj, timed_out)."""
    worker_fn = _WORKER_FN[kind]
    rq = multiprocessing.Queue()
    p  = multiprocessing.Process(
        target=worker_fn,
        args=(copy.deepcopy(query), n_scenarios, extra_kwargs, rq)
    )
    p.start()
    p.join(timeout=TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, None, True
    result = rq.get()
    if len(result) == 3:          # error tuple
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

    print(f'Query       : {SQL_PATH}')
    print(f'Timeout/run : {TIMEOUT}s')
    print(f'Iterations  : {N_ITERATIONS}\n')

    # timed_out[label] = True once a method fails/times-out at any scenario count
    timed_out = {label: False for label, _, _ in METHODS}
    rows = []

    for n in SCENARIO_COUNTS:
        print(f'=== Optimisation scenarios: {n} ===')
        for label, kind, extra_kwargs in METHODS:
            if timed_out[label]:
                print(f'  [{label}] skipped (previously timed out or no solution found)')
                rows.append({'n': n, 'method': label,
                             'rt_avg': 'N/A', 'rt_std': 'N/A',
                             'ob_avg': 'N/A', 'ob_std': 'N/A',
                             'n_ok': 0, 'status': 'SKIPPED'})
                continue

            runtimes, objs = [], []
            final_status = 'OK'

            for it in range(1, N_ITERATIONS + 1):
                q = copy.deepcopy(base_query)
                print(f'  [{label}] iter {it}/{N_ITERATIONS}...', end='', flush=True)
                t0 = time.time()
                runtime, obj, did_timeout = _run_method(q, n, kind, extra_kwargs)
                elapsed = time.time() - t0

                if did_timeout:
                    print(f' TIMEOUT after {elapsed:.1f}s — stopping iterations')
                    timed_out[label] = True
                    final_status = 'TIMEOUT'
                    break
                elif runtime is None:
                    print(' ERROR')
                    final_status = 'ERROR'
                    timed_out[label] = True
                    break
                elif obj is None:
                    print(f' runtime={runtime:.2f}s  obj=None — no solution, stopping')
                    timed_out[label] = True
                    final_status = 'NO_SOLUTION'
                    break
                else:
                    obj_str = f'{obj:.4f}'
                    print(f' runtime={runtime:.2f}s  obj={obj_str}')
                    runtimes.append(runtime)
                    objs.append(obj)

            rt_avg, rt_std = _fmt_avg_std(runtimes)
            ob_avg, ob_std = _fmt_avg_std(objs)
            rows.append({'n': n, 'method': label,
                         'rt_avg': rt_avg, 'rt_std': rt_std,
                         'ob_avg': ob_avg, 'ob_std': ob_std,
                         'n_ok': len(runtimes), 'status': final_status})
        print()

    col_w  = 26
    header = (f'{"Scenarios":>10}  {"Method":<{col_w}}  '
              f'{"AvgRuntime(s)":>14}  {"StdRuntime(s)":>14}'
              f'  {"AvgObj":>12}  {"StdObj":>12}  {"N/Iter":>6}  {"Status":>11}')
    sep    = '-' * len(header)
    lines  = [header, sep]
    for r in rows:
        lines.append(
            f'{r["n"]:>10}  {r["method"]:<{col_w}}  '
            f'{r["rt_avg"]:>14}  {r["rt_std"]:>14}'
            f'  {r["ob_avg"]:>12}  {r["ob_std"]:>12}  {r["n_ok"]:>6}  {r["status"]:>11}'
        )

    print('\n' + '\n'.join(lines))

    eval_dir = os.path.join(_PROJECT_ROOT, 'Evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, 'benchmark_iterative_constraint_addition.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
