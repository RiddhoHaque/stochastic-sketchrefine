#!/usr/bin/env python3
"""
Benchmark script: solve Q2.sql with RCLSolve using LP-first,
comparing use_pdhg=True (PDHG on GPU) vs use_pdhg=False (simplex).

Both runs use:
    solve_lp_first=True
    use_gpu=True          (so PDHG can activate when use_pdhg=True)

The number of optimisation scenarios is fixed at N_SCENARIOS.  Each
configuration is run N_ITERATIONS times; average and standard deviation of
runtime and objective value are reported.  Each run is capped at TIMEOUT
seconds.  If any iteration times out, remaining iterations for that
configuration are skipped.

Run from the project root:
    python DemoScripts/benchmark_use_pdhg.py
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
from Hyperparameters.Hyperparameters import Hyperparameters
from StochasticPackageQuery.Parser.Parser import Parser

SQL_PATH     = os.path.join(_PROJECT_ROOT, 'Workloads', 'DemoWorkload', 'Q2.sql')
N_SCENARIOS  = 100
TIMEOUT      = 20 * 60  # seconds
N_ITERATIONS = 5

CONFIGS = [
    ('LP_First / use_pdhg=True',
     {'solve_lp_first': True, 'use_gpu': True, 'use_pdhg': True}),
    ('LP_First / use_pdhg=False',
     {'solve_lp_first': True, 'use_gpu': True, 'use_pdhg': False}),
]


def _worker(query, extra_kwargs, result_queue):
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
            **extra_kwargs
        )
        package, obj_val = solver.solve()
        runtime = solver.get_metrics().get_runtime()
        result_queue.put((runtime, obj_val if package is not None else None))
    except Exception as exc:
        result_queue.put((None, None, str(exc)))


def _run(query, extra_kwargs):
    """Run solver in a subprocess; return (runtime, obj, timed_out)."""
    rq = multiprocessing.Queue()
    p  = multiprocessing.Process(target=_worker,
                                 args=(copy.deepcopy(query), extra_kwargs, rq))
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


def _fmt_avg_std(values):
    """Return (avg_str, std_str) for a list of floats, or ('N/A', 'N/A') if empty."""
    if not values:
        return 'N/A', 'N/A'
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f'{avg:.4f}', f'{std:.4f}'


def main():
    with open(SQL_PATH, 'r') as f:
        base_query = Parser().parse(f.readlines())

    print(f'Query       : {SQL_PATH}')
    print(f'N scenarios : {N_SCENARIOS}')
    print(f'Timeout/run : {TIMEOUT}s')
    print(f'Iterations  : {N_ITERATIONS}\n')

    rows = []
    for label, extra_kwargs in CONFIGS:
        runtimes, objs = [], []
        final_status = 'OK'

        for it in range(1, N_ITERATIONS + 1):
            print(f'  [{label}] iter {it}/{N_ITERATIONS}...', end='', flush=True)
            t0 = time.time()
            runtime, obj, did_timeout = _run(base_query, extra_kwargs)
            elapsed = time.time() - t0

            if did_timeout:
                print(f' TIMEOUT after {elapsed:.1f}s — stopping iterations')
                final_status = 'TIMEOUT'
                break
            elif runtime is None:
                print(' ERROR')
                final_status = 'ERROR'
            else:
                obj_str = f'{obj:.4f}' if obj is not None else 'None'
                print(f' runtime={runtime:.2f}s  obj={obj_str}')
                runtimes.append(runtime)
                if obj is not None:
                    objs.append(obj)

        rt_avg, rt_std = _fmt_avg_std(runtimes)
        ob_avg, ob_std = _fmt_avg_std(objs)
        rows.append({
            'label': label,
            'rt_avg': rt_avg, 'rt_std': rt_std,
            'ob_avg': ob_avg, 'ob_std': ob_std,
            'n_ok': len(runtimes), 'status': final_status,
        })

    col_w  = 40
    header = (f'{"Configuration":<{col_w}}  {"AvgRuntime(s)":>14}  {"StdRuntime(s)":>14}'
              f'  {"AvgObj":>12}  {"StdObj":>12}  {"N/Iter":>6}  {"Status":>8}')
    sep    = '-' * len(header)
    lines  = [header, sep]
    for r in rows:
        lines.append(
            f'{r["label"]:<{col_w}}  {r["rt_avg"]:>14}  {r["rt_std"]:>14}'
            f'  {r["ob_avg"]:>12}  {r["ob_std"]:>12}  {r["n_ok"]:>6}  {r["status"]:>8}'
        )

    print('\n' + '\n'.join(lines))

    eval_dir = os.path.join(_PROJECT_ROOT, 'Evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, 'benchmark_use_pdhg.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
