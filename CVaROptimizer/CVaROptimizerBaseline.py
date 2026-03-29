import copy
import numpy as np
import gurobipy as gp

from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from SketchRefine.SketchRefine import SketchRefine
from StochasticPackageQuery.Query import Query
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from Utils.GurobiLicense import GurobiLicense
from Utils.ObjectiveType import ObjectiveType
from Utils.TailType import TailType
from Validator.Validator import Validator


class CVaROptimizerBaseline:

    def __init__(self, query: Query, dbInfo: DbInfo,
                 num_val_scenarios: int):
        self.__query = query
        self.__dbInfo = dbInfo
        self.__num_validation_scenarios = num_val_scenarios
        self.__gurobi_env = gp.Env(params=GurobiLicense.OPTIONS)
        self.__gurobi_env.setParam('OutputFlag', 0)

    def __make_cvar_objective_constraint(
        self, attribute: str, v: float,
        tail_type: TailType, probability: float,
        is_maximize: bool
    ) -> CVaRConstraint:
        cvar = CVaRConstraint()
        cvar.set_attribute_name(attribute)
        cvar.set_inequality_sign('>' if is_maximize else '<')
        cvar.set_sum_limit(v)
        cvar.set_percentage_of_scenarios(probability * 100)
        cvar.set_tail_type('l' if tail_type == TailType.LOWEST else 'h')
        return cvar

    def __solve_feasibility(self, query: Query):
        """Run SketchRefine in check_feasibility mode and return (package, obj)."""
        solver = SketchRefine(
            query=query, dbInfo=self.__dbInfo,
            is_lp_relaxation=False, check_feasibility=True,
            gurobi_env=self.__gurobi_env
        )
        return solver.solve()

    def solve(self):
        objective = self.__query.get_objective()
        assert objective.is_cvar_objective()

        attr        = objective.get_attribute_name()
        tail_type   = objective.get_tail_type()
        probability = objective.get_percentage_of_scenarios()   # 0–1
        is_maximize = (objective.get_objective_type() == ObjectiveType.MAXIMIZATION)

        # Step 1: Solve without the CVaR objective constraint → base bound
        base_package, _ = self.__solve_feasibility(self.__query)
        if base_package is None:
            return (None, 0.0)

        validator = Validator(
            query=self.__query,
            dbInfo=self.__dbInfo,
            no_of_validation_scenarios=self.__num_validation_scenarios
        )
        base_cvar = validator.get_cvar_constraint_satisfaction(
            package_dict=base_package,
            cvar_constraint=self.__make_cvar_objective_constraint(
                attr, 0.0, tail_type, probability, is_maximize
            )
        )
        print(f'[CVaR Binary Search] Base CVaR: {base_cvar}')

        # Step 2: Find the outer bound by doubling (MAXIMIZE) or halving (MINIMIZE)
        # until infeasible
        v_bound = base_cvar
        while True:
            if is_maximize:
                v_bound += abs(v_bound) if v_bound != 0 else 1.0
            else:
                v_bound -= abs(v_bound) if v_bound != 0 else 1.0
            extra_cvar = self.__make_cvar_objective_constraint(
                attr, v_bound, tail_type, probability, is_maximize)

            query = copy.deepcopy(self.__query)
            query.add_constraint(extra_cvar)

            test_package, _ = self.__solve_feasibility(query)

            if test_package is None:
                print(f'[CVaR Binary Search] Outer bound found at V={v_bound:.4f} (infeasible)')
                break
            print(f'[CVaR Binary Search] Still feasible at V={v_bound:.4f}, expanding...')

        # Step 3: Set binary search range
        if is_maximize:
            low, high = base_cvar, v_bound
        else:
            low, high = v_bound, base_cvar

        best_package = base_package
        iteration    = 0

        # Step 4: Binary search
        while ((high - low) / (np.abs(low) + 1e-10)) > Hyperparameters.APPROXIMATION_BOUND:
            v_mid = (low + high) / 2
            print(f'[CVaR Binary Search] Iteration {iteration + 1}: trying V={v_mid:.4f}')
            iteration += 1
            extra_cvar = self.__make_cvar_objective_constraint(
                attr, v_mid, tail_type, probability, is_maximize)

            query = copy.deepcopy(self.__query)
            query.add_constraint(extra_cvar)

            package, _ = self.__solve_feasibility(query)

            if package is not None:
                best_package = package
                print(f'[CVaR Binary Search]   Feasible at V={v_mid:.4f}')
                if is_maximize:
                    low = v_mid
                else:
                    high = v_mid
            else:
                print(f'[CVaR Binary Search]   Infeasible at V={v_mid:.4f}')
                if is_maximize:
                    high = v_mid
                else:
                    low = v_mid

        # Step 5: Return best package with its CVaR bound
        if is_maximize:
            return (best_package, low)
        else:
            return (best_package, high)
