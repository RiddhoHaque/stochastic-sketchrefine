#!/usr/bin/env python3
"""
For every GARCH_Portfolio_YYYY relation (years 2017, 2019, 2022, 2023, 2025):

  1. Solve Workloads/DemoWorkload/Q1.sql with SketchRefine.
     Print the chosen package (with full attribute values) and its BackTest gain.

  2. Build an alternative CVaR-maximisation query:
         SUM(Price)        <= 5000
         EXPECTED SUM(Gain) >= 0.90 * O      (O = SketchRefine objective value)
         MAXIMIZE  EXPECTED SUM(Gain)  IN LOWER 0.05 TAIL
     Solve it with:
       a) CVaROptimizerBaseline
       b) SketchRefine
     Print each package + BackTest gain.

Run from the project root or from DemoScripts:
    python DemoScripts/QueryRunner.py
"""
import os
import sys
import time

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import copy
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

import psycopg2

from BackTest.BackTest import BackTest
from CVaROptimizer.CVaROptimizerBaseline import CVaROptimizerBaseline
from DbInfo.GBMPortfolioInfo import GBMPortfolioInfo
from DbInfo.GarchPortfolioInfo import GarchPortfolioInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from SketchRefine.SketchRefine import SketchRefine
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from StochasticPackageQuery.Objective.Objective import Objective
from StochasticPackageQuery.Parser.Parser import Parser
from StochasticPackageQuery.Query import Query

SQL_PATH = os.path.join(_PROJECT_ROOT, 'Workloads', 'DemoWorkload', 'Q1.sql')
YEARS    = range(2021, 2026)
TIMEOUT  = 60 * 60   # seconds per solver call

_INDEX_LABELS = {
    'dow_pct':         'DOW',
    'sp500_pct':       'S&P500',
    'nasdaq_pct':      'NASDAQ',
    'russell2000_pct': 'Russell2000',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relation_exists(relation: str) -> bool:
    try:
        PgConnection.Execute(f'SELECT 1 FROM {relation} LIMIT 1')
        PgConnection.Fetch()
        return True
    except psycopg2.errors.UndefinedTable:
        PgConnection.CONNECTION.rollback()
        return False
    except Exception:
        if PgConnection.CONNECTION is not None:
            PgConnection.CONNECTION.rollback()
        return False


def _print_package_tuples(relation: str, package: dict):
    """Fetch all attributes for the package tuples and print a table."""
    ids_array = '{' + ','.join(str(int(tid)) for tid in package) + '}'
    PgConnection.Execute(
        f"SELECT * FROM {relation} "
        f"WHERE id = ANY('{ids_array}'::int[]) ORDER BY id;"
    )
    col_names = [desc[0] for desc in PgConnection.CURSOR.description]
    rows      = PgConnection.Fetch()
    row_by_id = {int(row[col_names.index('id')]): row for row in rows}

    col_w    = 16
    all_cols = ['multiplicity'] + col_names
    header   = '  ' + '  '.join(f'{c:>{col_w}}' for c in all_cols)
    print(header)
    print('  ' + '-' * (col_w + 2) * len(all_cols))

    for tid, mult in sorted(package.items()):
        if tid not in row_by_id:
            print(f'  {"?":>{col_w}}  {tid:>{col_w}}  (not found)')
            continue
        row    = row_by_id[tid]
        values = [f'{mult:>{col_w}}'] + [f'{v:>{col_w}}' for v in row]
        print('  ' + '  '.join(values))


def _build_cvar_query(base_query: Query, relation: str,
                      min_expected_gain: float) -> Query:
    """
    Build the alternative query:
        SELECT Package(*) FROM <relation>
        WHERE <base predicate in base_query>
        SUCH THAT
            <all deterministic constraints from base_query>  AND
            EXPECTED SUM(gain) >= min_expected_gain
        MAXIMIZE EXPECTED SUM(gain) IN LOWER 0.05 TAIL
    """
    q = Query()
    q.set_projected_attributes(base_query.get_projected_attributes())
    q.set_relation(relation)
    q.set_base_predicate(base_query.get_base_predicate())

    # Copy deterministic (non-stochastic) constraints
    for c in base_query.get_constraints():
        if c.is_deterministic_constraint():
            q.add_constraint(copy.deepcopy(c))
        if c.is_package_size_constraint():
            q.add_constraint(copy.deepcopy(c))
        if c.is_repeat_constraint():
            q.add_constraint(copy.deepcopy(c))

    # Expected-sum floor: EXPECTED SUM(gain) >= 0.90 * O
    esc = ExpectedSumConstraint()
    esc.set_attribute_name('gain')
    esc.set_inequality_sign('>')
    esc.set_sum_limit(min_expected_gain)
    q.add_constraint(esc)

    # Objective: MAXIMIZE EXPECTED SUM(gain) IN LOWER 0.05 TAIL
    obj = Objective()
    obj.set_objective_type(True)   # MAXIMIZE
    obj.set_stochasticity(True)    # EXPECTED (stochastic)
    obj.set_attribute_name('gain')
    obj.set_as_cvar()
    obj.set_tail_type('l')         # LOWER
    for c in base_query.get_constraints():
        if c.is_cvar_constraint():
            # CVaRConstraint stores percentage as 0-100; Objective expects 0-1
            tail_fraction = c.get_percentage_of_scenarios() / 100.0
            tail_str = str(tail_fraction)
            for t in tail_str:
                obj.add_character_to_percentage_of_scenarios(t)
            break
    q.set_objective(obj)

    return q


def _solve_and_report(label: str, solver_fn, relation: str) -> dict:
    """
    Call solver_fn() → (package, objective_value).
    Print result, run BackTest, return summary dict.
    """
    print(f'\n  [{label}]')
    package, obj_val, runtime = solver_fn()

    if package is None:
        print('    No feasible package found.')
        return {
            'label': label, 'relation': relation,
            'objective_value': None, 'runtime': None,
            'actual_gain': None,
            'dow_pct': None, 'sp500_pct': None,
            'nasdaq_pct': None, 'russell2000_pct': None,
            'total_investment': None, 'package_size': 0,
        }

    print(f'    Objective value : {obj_val}')
    print(f'    Package ({len(package)} tuple(s)):')
    _print_package_tuples(relation, package)

    bt          = BackTest(relation)
    actual_gain = bt.get_gain(package)
    bench       = bt.get_benchmark_comparison(package)
    invest      = bench['total_investment']
    print(f'    Actual gain (BackTest): {actual_gain:.4f}')
    print(f'    Total investment (SUM Price*Mult): {invest:.4f}')
    for key, label_str in _INDEX_LABELS.items():
        pct = bench.get(key)
        if pct is not None:
            print(f'    {label_str} profit (same window, same investment): {invest * pct / 100.0:.4f}')
        else:
            print(f'    {label_str} profit (same window, same investment): N/A')

    return {
        'label': label, 'relation': relation,
        'objective_value': obj_val,
        'runtime': runtime,
        'actual_gain': actual_gain,
        'dow_pct':          bench.get('dow_pct'),
        'sp500_pct':        bench.get('sp500_pct'),
        'nasdaq_pct':       bench.get('nasdaq_pct'),
        'russell2000_pct':  bench.get('russell2000_pct'),
        'total_investment': invest,
        'package_size': len(package),
    }


def _run_one(base_query: Query, relation: str, db_info) -> list[dict]:
    """
    Run all three solvers for one relation.
    Returns a list of summary dicts (one per solver).
    """
    results = []

    # ------------------------------------------------------------------
    # Phase 1: SketchRefine on original Q1
    # ------------------------------------------------------------------
    print('\n  [SketchRefine / Q1]')
    base_query.set_relation(relation)
    solver  = SketchRefine(query=base_query, dbInfo=db_info, is_lp_relaxation=False, early_termination=True)
    package, _ = solver.solve()
    metric  = solver.get_metrics()
    obj_val = metric.get_objective_value()

    sr_result = {
        'label': 'SketchRefine/Q1', 'relation': relation,
        'objective_value': obj_val,
        'runtime': metric.get_runtime(),
        'actual_gain': None,
        'dow_pct': None, 'sp500_pct': None,
        'nasdaq_pct': None, 'russell2000_pct': None,
        'total_investment': None, 'package_size': 0,
        'timed_out': False,
    }

    if package is None:
        print('    No feasible package found.')
        results.append(sr_result)
        return results   # nothing to pivot on without O

    sr_result['package_size'] = len(package)
    print(f'    Objective value : {obj_val}')
    print(f'    Runtime (s)     : {metric.get_runtime():.3f}')
    print(f'    Package ({len(package)} tuple(s)):')
    _print_package_tuples(relation, package)

    bt          = BackTest(relation)
    actual_gain = bt.get_gain(package)
    bench       = bt.get_benchmark_comparison(package)
    invest      = bench['total_investment']
    sr_result['actual_gain']      = actual_gain
    sr_result['total_investment'] = invest
    for key in _INDEX_LABELS:
        sr_result[key] = bench.get(key)
    print(f'    Actual gain (BackTest): {actual_gain:.4f}')
    print(f'    Total investment (SUM Price*Mult): {invest:.4f}')
    for key, label_str in _INDEX_LABELS.items():
        pct = bench.get(key)
        if pct is not None:
            print(f'    {label_str} profit (same window, same investment): {invest * pct / 100.0:.4f}')
        else:
            print(f'    {label_str} profit (same window, same investment): N/A')
    results.append(sr_result)

    # ------------------------------------------------------------------
    # Phase 2: Alternative CVaR query with threshold = 0.50 * O
    # ------------------------------------------------------------------
    min_gain    = min(0.50 * obj_val, obj_val - 20.0)
    cvar_query  = _build_cvar_query(base_query, relation, min_gain)
    print(f'\n  Building CVaR query with EXPECTED SUM(Gain) >= {min_gain:.4f}')

    # SketchRefine on CVaR query (L-CVaR)
    def _run_lcvar():
        q = copy.deepcopy(cvar_query)
        solver = SketchRefine(query=q, dbInfo=db_info, is_lp_relaxation=False, optimize_lcvar=True, early_termination=True)
        #start_time = time.time()
        #solver = CVaROptimizerBaseline(query=q, dbInfo=db_info, num_val_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS)
        pkg, _ = solver.solve()
        runtime = solver.get_metrics().get_runtime()
        #runtime = time.time() - start_time
        return pkg, (solver.get_metrics().get_objective_value() if pkg is not None else 0.0), runtime

    lc = _solve_and_report('SketchRefine/CVaR(L-CVaR)', _run_lcvar, relation)
    lc['timed_out'] = False
    results.append(lc)
    return results


def _run_one_worker(base_query: Query, relation: str, db_info,
                    result_queue: multiprocessing.Queue):
    results = _run_one(base_query, relation, db_info)
    result_queue.put(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _format_summary_table(all_summaries: list[dict]) -> list[str]:
    """Format the summary table and return lines to be printed/saved."""
    lines = []
    _idx_cols = list(_INDEX_LABELS.items())   # [(key, label), ...]
    col_w     = 14
    hdr = (f'{"Relation":<30}  {"Solver":<28}  {"Obj Value":>12}  '
           f'{"Runtime(s)":>12}  {"Pkg Size":>9}  {"Actual Gain":>14}  ' +
           '  '.join(f'{lbl + " Profit":>{col_w}}' for _, lbl in _idx_cols))
    lines.append(hdr)
    lines.append('-' * len(hdr))

    for s in all_summaries:
        if s.get('timed_out'):
            lines.append(f'{s["relation"]:<30}  {"ALL":<28}  {"TIMEOUT":>12}')
            continue
        slabel   = s.get('label', '')
        obj_str  = f'{s["objective_value"]:.4f}' if s['objective_value'] is not None else 'N/A'
        rt_str   = f'{s["runtime"]:.3f}'          if s['runtime']         is not None else 'N/A'
        gain_str = f'{s["actual_gain"]:.4f}'      if s['actual_gain']     is not None else 'N/A'
        invest   = s.get('total_investment')
        idx_strs = []
        for key, _ in _idx_cols:
            pct = s.get(key)
            if invest is not None and pct is not None:
                idx_strs.append(f'{invest * pct / 100.0:{col_w}.4f}')
            else:
                idx_strs.append(f'{"N/A":>{col_w}}')
        lines.append(
            f'{s["relation"]:<30}  {slabel:<28}  {obj_str:>12}  '
            f'{rt_str:>12}  {s["package_size"]:>9}  {gain_str:>14}  ' +
            '  '.join(idx_strs)
        )
    return lines


def main():
    with open(SQL_PATH, 'r') as f:
        base_query = Parser().parse(f.readlines())

    print(f'Query file: {SQL_PATH}')
    print()

    all_summaries = []   # flat list of result dicts

    # Run both GBM and GARCH relations
    for portfolio_type, db_info_class in [#('GBM_Portfolio', GBMPortfolioInfo),
                                           ('GARCH_Portfolio', GarchPortfolioInfo)]:
        print('=' * 72)
        print(f'  {portfolio_type}')
        print('=' * 72)

        for year in YEARS:
            relation = f'{portfolio_type}_{year}'
            print(f'\n--- {relation} ---')

            if not _relation_exists(relation):
                print('  Relation not found — skipping.')
                continue

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_run_one_worker,
                args=(base_query, relation, db_info_class, result_queue)
            )
            process.start()
            process.join(timeout=TIMEOUT)

            if process.is_alive():
                process.terminate()
                process.join()
                print(f'  Timed out after {TIMEOUT}s — killed.')
                all_summaries.append({
                    'label': 'ALL', 'relation': relation,
                    'objective_value': None, 'runtime': None,
                    'actual_gain': None, 'dow_pct': None, 'sp500_pct': None,
                    'package_size': 0, 'timed_out': True,
                })
            else:
                for r in result_queue.get():
                    r.setdefault('timed_out', False)
                    all_summaries.append(r)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print('=' * 72)
    print('  SUMMARY')
    print('=' * 72)

    # Format and print summary
    summary_lines = _format_summary_table(all_summaries)
    for line in summary_lines:
        print(line)

    # Write summary to file
    eval_dir = os.path.join(_PROJECT_ROOT, 'Evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    output_file = os.path.join(eval_dir, 'summary_results.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f'\nSummary results saved to: {output_file}')


if __name__ == '__main__':
    main()
