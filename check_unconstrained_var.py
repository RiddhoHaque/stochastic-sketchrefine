"""
Usage: python check_unconstrained_var.py <portfolio|tpch> <query_number>

For each of 1x and 20x variance, finds the probabilistically unconstrained
package and prints its VaR (and actual P) for every VaR constraint in the query.
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from CVaRification.RCLSolve import RCLSolve
from DbInfo.PortfolioInfo import PortfolioInfo
from DbInfo.TpchInfo import TpchInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from StochasticPackageQuery.Parser.Parser import Parser
from Utils.RelationalOperators import RelationalOperators
from Validator.Validator import Validator

db = sys.argv[1].lower()
query_num = int(sys.argv[2])

if db == 'portfolio':
    relation_1x  = 'Stock_Investments_Volatility_1x'
    relation_20x = 'Stock_Investments_Volatility_20x'
    query_file   = f'Workloads/PortfolioWorkload/Q{query_num}.sql'
    dbInfo       = PortfolioInfo
elif db == 'tpch':
    relation_1x  = 'Lineitem_Variance_1x'
    relation_20x = 'Lineitem_Variance_20x'
    query_file   = f'Workloads/TpchWorkload/Q{query_num}.sql'
    dbInfo       = TpchInfo
else:
    print(f'Unknown database: {db}. Use "portfolio" or "tpch".')
    sys.exit(1)

with open(query_file) as f:
    query_lines = f.readlines()

print(f'Query file : {query_file}')

for label, relation in [('1x', relation_1x), ('20x', relation_20x)]:
    query = Parser().parse(query_lines)
    query.set_relation(relation)

    solver = RCLSolve(
        query=query,
        linear_relaxation=False,
        dbInfo=dbInfo,
        init_no_of_scenarios=Hyperparameters.INIT_NO_OF_SCENARIOS,
        no_of_validation_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
        approximation_bound=Hyperparameters.APPROXIMATION_BOUND,
        sampling_tolerance=Hyperparameters.SAMPLING_TOLERANCE,
        bisection_threshold=Hyperparameters.BISECTION_THRESHOLD,
        max_opt_scenarios=Hyperparameters.MAX_OPT_SCENARIOS_IN_PRACTICE,
    )

    # Run unconstrained LP (no probabilistic constraints)
    solver._RCLSolve__model_setup(
        no_of_scenarios=Hyperparameters.INIT_NO_OF_SCENARIOS,
        no_of_scenarios_to_consider=[],
        probabilistically_constrained=False,
    )
    pkg = solver._RCLSolve__get_package()

    print(f'\n=== {label} | {relation} ===')
    print(f'Unconstrained package : {pkg}')

    if pkg is not None and len(pkg) > 0:
        ids = solver._RCLSolve__ids
        values = solver._RCLSolve__values
        id_to_idx = {id_: i for i, id_ in enumerate(ids)}
        for attr, vals in values.items():
            total = sum(vals[id_to_idx[id_]] * qty for id_, qty in pkg.items())
            print(f'  SUM({attr})            : {total:.4f}')

    if pkg is None or len(pkg) == 0:
        print('  (empty / infeasible — skipping VaR computation)')
        continue

    validator = Validator(query, dbInfo, Hyperparameters.NO_OF_VALIDATION_SCENARIOS)

    for constraint in query.get_constraints():
        if constraint.is_var_constraint():
            attr  = constraint.get_attribute_name()
            limit = constraint.get_sum_limit()
            prob  = constraint.get_probability_threshold()
            sign  = constraint.get_inequality_sign()
            if sign == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                sign_str = '>='
            elif sign == RelationalOperators.LESS_THAN_OR_EQUAL_TO:
                sign_str = '<='
            else:
                sign_str = '='

            # Compute VaR from the correct tail for the constraint direction.
            # >= constraint: p-th percentile from top (highest scenarios).
            # <= constraint: p-th percentile from bottom (lowest scenarios).
            scenarios_raw = validator._Validator__get_scenarios_and_ids(pkg, attr)
            mat = np.array(scenarios_raw[0])
            mults = np.array([m for _, m in scenarios_raw[1]])
            scores = mat.T @ mults
            n = len(scores)
            k = min(int(np.floor(prob * n)), n - 1)
            if sign == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                var = float(np.sort(scores)[::-1][k])
            else:
                var = float(np.sort(scores)[k])

            actual_p = validator.get_var_constraint_satisfaction(pkg, constraint)
            feasible = actual_p >= prob
            print(f'  Constraint : SUM({attr}) {sign_str} {limit} WITH P >= {prob}')
            print(f'    VaR at p={prob} : {var:.4f}')
            print(f'    Actual P       : {actual_p:.4f}  {"PASS" if feasible else "FAIL"}')
