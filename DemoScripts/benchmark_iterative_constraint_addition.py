#!/usr/bin/env python3
"""
Benchmark script 5: runtime and objective value for Q2.sql as the number of
optimisation scenarios increases.

Three solver configurations are compared:
  1. Default RCLSolve                       (no extra flags)
  2. RCLSolve with iterative_constraint_addition=True
  3. RCLSolve with iterative_constraint_addition=True AND solve_lp_first=True

The number of optimisation scenarios is varied over [10, 20, 50, 80, 100].
Each solver run is capped at TIMEOUT seconds.  If a solver times out at a
given scenario count it is not attempted at higher counts.

RCLSolve is configured so it cannot increase the scenario count beyond the
value under test (max_opt_scenarios = init_no_of_scenarios = N).

Run from the project root:
    python DemoScripts/benchmark_iterative_constraint_addition.py
"""
import multiprocessing
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from CVaRification.RCLSolve import RCLSolve
from DbInfo.GarchPortfolioInfo import GarchPortfolioInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from StochasticPackageQuery.Parser.Parser import Parser

SQL_PATH        = os.path.join(_PROJECT_ROOT, 'Workloads', 'DemoWorkload', 'Q2.sql')
SCENARIO_COUNTS = [10, 20, 50, 80, 100]
TIMEOUT         = 20 * 60   # seconds

METHODS = [
    ('Default_RCLSolve',       {}),
    ('Iterative_Z',            {'iterative_constraint_addition': True}),
    ('Iterative_Z+LP_First',   {'iterative_constraint_addition': True,
                                'solve_lp_first': True}),
]


def _worker(query, n_scenarios, extra_kwargs, result_queue):
    """Worker process: instantiate and run RCLSolve, put (runtime, obj) in queue."""
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


def _run_method(query, n_scenarios, extra_kwargs):
    """Run a solver in a subprocess with timeout. Returns (runtime, obj, timed_out)."""
    rq = multiprocessing.Queue()
    p  = multiprocessing.Process(target=_worker,
                                 args=(query, n_scenarios, extra_kwargs, rq))
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


def main():
    with open(SQL_PATH, 'r') as f:
        base_query = Parser().parse(f.readlines())

    print(f'Query: {SQL_PATH}')
    print(f'Timeout per run: {TIMEOUT}s\n')

    # timed_out[method_label] = True if the method timed out at some N
    timed_out = {label: False for label, _ in METHODS}

    rows = []   # list of dicts for the summary table

    for n in SCENARIO_COUNTS:
        print(f'=== Optimisation scenarios: {n} ===')
        for label, extra_kwargs in METHODS:
            if timed_out[label]:
                print(f'  [{label}] skipped (previously timed out)')
                rows.append({'n': n, 'method': label,
                             'runtime': None, 'obj': None, 'status': 'SKIPPED'})
                continue

            import copy
            q = copy.deepcopy(base_query)
            print(f'  [{label}] running...', end='', flush=True)
            t0 = time.time()
            runtime, obj, did_timeout = _run_method(q, n, extra_kwargs)
            elapsed = time.time() - t0

            if did_timeout:
                print(f' TIMEOUT after {elapsed:.1f}s')
                timed_out[label] = True
                rows.append({'n': n, 'method': label,
                             'runtime': None, 'obj': None, 'status': 'TIMEOUT'})
            elif runtime is None:
                print(' ERROR')
                rows.append({'n': n, 'method': label,
                             'runtime': None, 'obj': None, 'status': 'ERROR'})
            else:
                obj_str = f'{obj:.4f}' if obj is not None else 'None'
                print(f' runtime={runtime:.2f}s  obj={obj_str}')
                rows.append({'n': n, 'method': label,
                             'runtime': runtime, 'obj': obj, 'status': 'OK'})
        print()

    # Summary table
    col_w = 20
    header = (f'{"Scenarios":>10}  {"Method":<{col_w}}  '
              f'{"Runtime(s)":>12}  {"Obj Value":>14}  {"Status":>8}')
    sep    = '-' * len(header)
    lines  = [header, sep]
    for r in rows:
        rt_str  = f'{r["runtime"]:.2f}' if r['runtime'] is not None else 'N/A'
        obj_str = f'{r["obj"]:.4f}'     if r['obj']     is not None else 'N/A'
        lines.append(
            f'{r["n"]:>10}  {r["method"]:<{col_w}}  '
            f'{rt_str:>12}  {obj_str:>14}  {r["status"]:>8}'
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
