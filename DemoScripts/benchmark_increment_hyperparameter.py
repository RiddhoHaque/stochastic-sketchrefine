#!/usr/bin/env python3
"""
Benchmark script 6: runtime and objective value for Q2.sql as the
increment_in_number_of_constraints hyperparameter varies.

Solver configuration: RCLSolve with both
  iterative_constraint_addition=True  and  solve_lp_first=True.

The hyperparameter is varied over [1, 5, 10, 15, 20] with a fixed number of
optimisation scenarios (N_SCENARIOS).  Each run is capped at TIMEOUT seconds.

Run from the project root:
    python DemoScripts/benchmark_increment_hyperparameter.py
"""
import copy
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

SQL_PATH   = os.path.join(_PROJECT_ROOT, 'Workloads', 'DemoWorkload', 'Q2.sql')
N_SCENARIOS = 50
INCREMENTS  = [1, 5, 10, 15, 20]
TIMEOUT     = 20 * 60   # seconds


def _worker(query, increment, result_queue):
    """Worker process: run RCLSolve and put (runtime, obj) in queue."""
    try:
        solver = RCLSolve(
            query=query,
            linear_relaxation=False,
            dbInfo=GarchPortfolioInfo,
            init_no_of_scenarios=N_SCENARIOS,
            no_of_validation_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
            approximation_bound=Hyperparameters.APPROXIMATION_BOUND,
            sampling_tolerance=Hyperparameters.SAMPLING_TOLERANCE,
            bisection_threshold=Hyperparameters.BISECTION_THRESHOLD,
            max_opt_scenarios=N_SCENARIOS,
            iterative_constraint_addition=True,
            solve_lp_first=True,
            increment_in_number_of_constraints=increment,
        )
        package, obj_val = solver.solve()
        runtime = solver.get_metrics().get_runtime()
        result_queue.put((runtime, obj_val if package is not None else None))
    except Exception as exc:
        result_queue.put((None, None, str(exc)))


def _run(query, increment):
    """Run solver in subprocess with timeout. Returns (runtime, obj, timed_out)."""
    rq = multiprocessing.Queue()
    p  = multiprocessing.Process(target=_worker,
                                 args=(copy.deepcopy(query), increment, rq))
    p.start()
    p.join(timeout=TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, None, True
    result = rq.get()
    if len(result) == 3:
        print(f'  Error: {result[2]}')
        return None, None, False
    return result[0], result[1], False


def main():
    with open(SQL_PATH, 'r') as f:
        base_query = Parser().parse(f.readlines())

    print(f'Query         : {SQL_PATH}')
    print(f'N scenarios   : {N_SCENARIOS}')
    print(f'Timeout/run   : {TIMEOUT}s')
    print(f'Flags         : iterative_constraint_addition=True, solve_lp_first=True\n')

    rows = []
    for p_val in INCREMENTS:
        print(f'  [increment={p_val}] running...', end='', flush=True)
        t0 = time.time()
        runtime, obj, did_timeout = _run(base_query, p_val)
        elapsed = time.time() - t0

        if did_timeout:
            print(f' TIMEOUT after {elapsed:.1f}s')
            rows.append({'p': p_val, 'runtime': None, 'obj': None, 'status': 'TIMEOUT'})
        elif runtime is None:
            print(' ERROR')
            rows.append({'p': p_val, 'runtime': None, 'obj': None, 'status': 'ERROR'})
        else:
            obj_str = f'{obj:.4f}' if obj is not None else 'None'
            print(f' runtime={runtime:.2f}s  obj={obj_str}')
            rows.append({'p': p_val, 'runtime': runtime, 'obj': obj, 'status': 'OK'})

    # Summary table
    header = (f'{"Increment":>10}  {"Runtime(s)":>12}  {"Obj Value":>14}  {"Status":>8}')
    sep    = '-' * len(header)
    lines  = [header, sep]
    for r in rows:
        rt_str  = f'{r["runtime"]:.2f}' if r['runtime'] is not None else 'N/A'
        obj_str = f'{r["obj"]:.4f}'     if r['obj']     is not None else 'N/A'
        lines.append(
            f'{r["p"]:>10}  {rt_str:>12}  {obj_str:>14}  {r["status"]:>8}'
        )

    print('\n' + '\n'.join(lines))

    eval_dir = os.path.join(_PROJECT_ROOT, 'Evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, 'benchmark_increment_hyperparameter.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
