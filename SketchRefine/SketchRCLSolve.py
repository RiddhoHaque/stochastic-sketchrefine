import copy
import subprocess
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def _make_constr(lhs, sense, rhs):
    """Build a Gurobi TempConstr from (lhs, sense, rhs).

    Gurobi 11+ removed the positional addConstr(lhs, sense, rhs) form.
    """
    if sense == GRB.LESS_EQUAL:
        return lhs <= rhs
    if sense == GRB.GREATER_EQUAL:
        return lhs >= rhs
    return lhs == rhs

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False


def _has_gpu() -> bool:
    """Return True if a CUDA-capable GPU is available."""
    if _TORCH_AVAILABLE:
        return _torch.cuda.is_available()
    try:
        r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        pass
    return False

from CVaRification.CVaRificationSearchResults import CVaRificationSearchResults
from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OptimizationMetrics.OptimizationMetrics import OptimizationMetrics
from PgConnection.PgConnection import PgConnection
from SeedManager.SeedManager import SeedManager
from SketchRefine.SketchValidator import SketchValidator
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraint import VaRConstraint
from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraint import DeterministicConstraint
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from StochasticPackageQuery.Constraints.PackageSizeConstraint.PackageSizeConstraint import PackageSizeConstraint
from StochasticPackageQuery.Objective.Objective import Objective
from StochasticPackageQuery.Query import Query
from Utils.GurobiLicense import GurobiLicense
from Utils.Heap import Heap
from Utils.ObjectiveType import ObjectiveType
from Utils.RelationalOperators import RelationalOperators
from Utils.Stochasticity import Stochasticity
from Utils.TailType import TailType
from ValueGenerator.ValueGenerator import ValueGenerator


class SketchRCLSolve:

    def __init__(
        self, query: Query,
        linear_relaxation: bool,
        scenarios: dict[str, list[list[float]]],
        values: dict[str, list[float]],
        no_of_variables: int,
        no_of_opt_scenarios: int,
        validator: SketchValidator,
        approximation_bound: float,
        sampling_tolerance: float,
        bisection_threshold: float,
        partition_sizes: list[int],
        duplicate_vector: list[int],
        partition_id_for_each_index: dict[int, int],
        check_feasibility: bool = False,
        optimize_lcvar: bool = False,
        gurobi_env = None,
        early_termination: bool = False,
        iterative_constraint_addition: bool = True,
        set_initial_constraints_randomly: bool = False,
        solve_lp_first: bool = True,
        increment_in_number_of_constraints: int | None = None,
        use_gpu: bool = True,
        solve_dual_lp: bool = True,
        use_pdhg: bool = False
    ):
        self.__query = copy.deepcopy(query)
        if gurobi_env is None:
            self.__gurobi_env = gp.Env(params=GurobiLicense.OPTIONS)
        else:
            self.__gurobi_env = gurobi_env
        self.__gurobi_env.setParam(
            'OutputFlag', 0
        )
        self.__model = gp.Model(
            env=self.__gurobi_env)
        self.__check_feasibility = check_feasibility
        self.__optimize_lcvar = optimize_lcvar
        self.__early_termination = early_termination
        if solve_lp_first:
            iterative_constraint_addition = True
        self.__iterative_constraint_addition = iterative_constraint_addition
        self.__set_initial_constraints_randomly = set_initial_constraints_randomly
        self.__solve_lp_first = solve_lp_first
        self.__increment_in_number_of_constraints = (
            increment_in_number_of_constraints
            if increment_in_number_of_constraints is not None
            else Hyperparameters.INCREMENT_IN_NUMBER_OF_CONSTRAINTS
        )
        self.__has_gpu = use_gpu and _has_gpu()
        self.__solve_dual_lp = solve_dual_lp
        self.__use_pdhg = use_pdhg
        self.__z_constrained_scenarios: set = set()
        self.__z_iterative_active = False
        if check_feasibility:
            self.__model.Params.SolutionLimit = 1
            self.__model.Params.MIPFocus = 1
        self.__is_linear_relaxation = \
            linear_relaxation
        
        self.__partition_sizes = \
            partition_sizes
        
        self.__validator = validator

        self.__approximation_bound = \
            approximation_bound
        self.__sampling_tolerance = \
            sampling_tolerance
        self.__bisection_threshold = \
            bisection_threshold
        
        self.__no_of_vars = no_of_variables
        
        self.__vars = []
        
        self.__risk_constraints = []
        self.__risk_to_lcvar_constraint_mapping = dict()
        self.__sorted_scenario_prefix_sums = {}
        self.__opt_scenario_scores_cache = {}

        self.__ids = [_ for _ in range(no_of_variables)]
        
        self.__metrics = OptimizationMetrics(
            'SketchRCLSolve',
            self.__is_linear_relaxation
        )

        self.__scenarios = scenarios
        self.__no_of_scenarios = no_of_opt_scenarios
        self.__values = values
        self.__duplicate_vector = duplicate_vector
        self.__partition_ids_for_each_index = partition_id_for_each_index
        self.__bounded_partition = None
        self.__partition_multiplicity_upper_bound = None

        if self.__query.get_objective().is_cvar_objective():
            self.__V = None
            self.__Zs = []
            self.__z_lazy_active = False  # kept for attribute existence; never set True

        print('Sketch Solver Initialized')

    def set_partition_upper_bound(self, partition_id, upper_bound):
        self.__bounded_partition = partition_id
        self.__partition_multiplicity_upper_bound = upper_bound

    def update_scenarios_and_values(
        self, updated_scenarios, updated_values,
        updated_duplicate_vector, updated_no_of_vars,
        partition_id_in_duplicate_vector 
    ):
        print('Changing alpha')
        self.__scenarios = updated_scenarios
        self.__values = updated_values
        self.__duplicate_vector = updated_duplicate_vector
        self.__no_of_vars = updated_no_of_vars
        self.__ids = [_ for _ in range(updated_no_of_vars)]
        self.__partition_ids_for_each_index = \
            partition_id_in_duplicate_vector
        self.__validator.update_partition_id_in_duplicate_vector(
            partition_id_in_duplicate_vector
        )
        self.__opt_scenario_scores_cache = {}
    
    def update_scenarios(
        self, updated_scenarios, no_of_scenarios
    ):
        self.__scenarios = updated_scenarios
        self.__no_of_scenarios = no_of_scenarios
        self.__opt_scenario_scores_cache = {}

    def __to_cvar_constraint(self, esc: ExpectedSumConstraint) -> CVaRConstraint:
        """Convert an ExpectedSumConstraint to an equivalent CVaRConstraint at 100% tail."""
        cvar = CVaRConstraint()
        cvar.set_attribute_name(esc.get_attribute_name())
        ineq = esc.get_inequality_sign()
        cvar.set_inequality_sign(
            '>' if ineq == RelationalOperators.GREATER_THAN_OR_EQUAL_TO else '<'
        )
        cvar.set_sum_limit(esc.get_sum_limit())
        cvar.set_percentage_of_scenarios(100.0)
        cvar.set_tail_type(
            'l' if ineq == RelationalOperators.GREATER_THAN_OR_EQUAL_TO else 'h'
        )
        return cvar

    def __is_searchable_stochastic_constraint(self, constraint) -> bool:
        return constraint.is_expected_sum_constraint() or \
            constraint.is_risk_constraint()

    def __get_searchable_constraints(self):
        return [
            constraint for constraint in self.__query.get_constraints()
            if self.__is_searchable_stochastic_constraint(constraint)
        ]

    def __get_upper_bound_for_repetitions(self) -> int:
        for constraint in self.__query.get_constraints():
            if constraint.is_repeat_constraint():
                return 1 + constraint.get_repetition_limit()
        return None
    
    
    def __add_variables_to_model(self, no_of_scenarios: int) -> None:
        max_repetition = \
            self.__get_upper_bound_for_repetitions()
        type = GRB.INTEGER
        if self.__is_linear_relaxation:
            type = GRB.CONTINUOUS
        self.__vars = []
        partition_no = 0
        for duplicates in self.__duplicate_vector:
            if max_repetition is not None:
                total_ub = max_repetition * \
                    self.__partition_sizes[partition_no]
            else:
                total_ub = None
            if self.__bounded_partition is not None and \
                    partition_no == self.__bounded_partition:
                bounded_ub = self.__partition_multiplicity_upper_bound
                total_ub = bounded_ub if total_ub is None \
                    else min(total_ub, bounded_ub)
            partition_no += 1
            for dup_idx in range(duplicates):
                if total_ub is not None:
                    ub = total_ub // duplicates
                    if dup_idx < total_ub % duplicates:
                        ub += 1
                    self.__vars.append(
                        self.__model.addVar(
                            lb=0, ub=ub, vtype=type
                        )
                    )
                else:
                    self.__vars.append(
                        self.__model.addVar(
                            lb=0, vtype=type
                        )
                    )

        if not self.__check_feasibility and not self.__optimize_lcvar and \
                self.__query.get_objective().is_cvar_objective():
            self.__V = self.__model.addVar(vtype=GRB.CONTINUOUS)
            self.__Zs = []
            for _ in range(no_of_scenarios):
                if self.__query.get_objective().get_tail_type() == \
                        TailType.LOWEST:
                    self.__Zs.append(
                        self.__model.addVar(ub=0, vtype=GRB.CONTINUOUS))
                else:
                    self.__Zs.append(
                        self.__model.addVar(lb=0, vtype=GRB.CONTINUOUS))
    

    def __get_gurobi_inequality(
            self, inequality_sign: RelationalOperators):
        if inequality_sign == RelationalOperators.EQUALS:
            return GRB.EQUAL
        if inequality_sign == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            return GRB.GREATER_EQUAL
        return GRB.LESS_EQUAL
    

    def __add_package_size_constraint_to_model(
        self, package_size_constraint: PackageSizeConstraint
    ):
        size_limit = \
            package_size_constraint.get_package_size_limit()
        gurobi_inequality = \
            self.__get_gurobi_inequality(
                package_size_constraint.get_inequality_sign())

        self.__model.addConstr(_make_constr(
            gp.LinExpr([1]*self.__no_of_vars, self.__vars),
            gurobi_inequality, size_limit
        ))
    

    def __add_deterministic_constraint_to_model(
        self, deterministic_constraint: DeterministicConstraint
    ):
        attribute = deterministic_constraint.get_attribute_name()
        gurobi_inequality = \
            self.__get_gurobi_inequality(
                deterministic_constraint.get_inequality_sign())
        sum_limit = deterministic_constraint.get_sum_limit()
        
        self.__model.addConstr(_make_constr(
            gp.LinExpr(self.__values[attribute], self.__vars),
            gurobi_inequality, sum_limit
        ))


    def __add_expected_sum_constraint_to_model(
        self, expected_sum_constraint: ExpectedSumConstraint,
        threshold_override: float | None = None
    ):
        attr = expected_sum_constraint.get_attribute_name()
        coefficients = []

        for idx in range(self.__no_of_vars):
            coefficients.append(
                np.average(self.__scenarios[attr][idx])
            )
        
        gurobi_inequality = self.__get_gurobi_inequality(
            expected_sum_constraint.get_inequality_sign()
        )

        expected_sum_limit = threshold_override
        if expected_sum_limit is None:
            expected_sum_limit = \
                expected_sum_constraint.get_sum_limit()

        return self.__model.addConstr(_make_constr(
            gp.LinExpr(coefficients, self.__vars),
            gurobi_inequality, expected_sum_limit
        ))
    

    def __get_no_of_scenarios_to_consider(
        self, cvar_constraint: CVaRConstraint,
        no_of_scenarios: int
    ):
        return max(1, int(np.floor(
            (cvar_constraint.get_percentage_of_scenarios()\
                *no_of_scenarios)/100)))


    def __preprocess_lcvar_prefix_sums(self) -> None:
        self.__sorted_scenario_prefix_sums = {}
        for constraint in self.__query.get_constraints():
            if not constraint.is_risk_constraint():
                continue
            attr = constraint.get_attribute_name()
            if constraint.is_cvar_constraint():
                tail_type = constraint.get_tail_type()
            else:
                if constraint.get_inequality_sign() == \
                        RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                    tail_type = TailType.LOWEST
                else:
                    tail_type = TailType.HIGHEST
            key = (attr, tail_type)
            if key in self.__sorted_scenario_prefix_sums:
                continue
            prefix_avgs = []
            for var_idx in range(self.__no_of_vars):
                arr = np.array(
                    self.__scenarios[attr][var_idx], dtype=np.float64)
                if tail_type == TailType.LOWEST:
                    sorted_arr = np.sort(arr)
                else:
                    sorted_arr = np.sort(arr)[::-1]
                k = np.arange(1, len(sorted_arr) + 1, dtype=np.float64)
                prefix_avgs.append(np.cumsum(sorted_arr) / k)
            self.__sorted_scenario_prefix_sums[key] = prefix_avgs
        if self.__optimize_lcvar and self.__query.get_objective().is_cvar_objective():
            obj = self.__query.get_objective()
            key = (obj.get_attribute_name(), obj.get_tail_type())
            if key not in self.__sorted_scenario_prefix_sums:
                attr = obj.get_attribute_name()
                tail_type = obj.get_tail_type()
                prefix_avgs = []
                for var_idx in range(self.__no_of_vars):
                    arr = np.array(
                        self.__scenarios[attr][var_idx], dtype=np.float64)
                    if tail_type == TailType.LOWEST:
                        sorted_arr = np.sort(arr)
                    else:
                        sorted_arr = np.sort(arr)[::-1]
                    k = np.arange(1, len(sorted_arr) + 1, dtype=np.float64)
                    prefix_avgs.append(np.cumsum(sorted_arr) / k)
                self.__sorted_scenario_prefix_sums[key] = prefix_avgs


    def __get_cvar_constraint_coefficients(
        self, cvar_constraint: CVaRConstraint,
        no_of_scenarios_to_consider: int,
        total_no_of_scenarios: int
    ):
        attr = cvar_constraint.get_attribute_name()
        key = (attr, cvar_constraint.get_tail_type())

        if key in self.__sorted_scenario_prefix_sums:
            k = no_of_scenarios_to_consider
            prefix_avgs = self.__sorted_scenario_prefix_sums[key]
            if k <= len(prefix_avgs[0]):
                return [prefix_avgs[var][k - 1]
                        for var in range(self.__no_of_vars)]

        tuple_wise_heaps = []
        is_max_heap = True
        if cvar_constraint.get_tail_type() == TailType.HIGHEST:
            is_max_heap = False

        for _ in range(self.__no_of_vars):
            tuple_wise_heaps.append(
                Heap(is_max_heap))

        for scenario_index in range(total_no_of_scenarios):
            for var in range(self.__no_of_vars):
                scenario_value = self.__scenarios[attr][var][
                    scenario_index]
                tuple_wise_heaps[var].push(scenario_value)
                if tuple_wise_heaps[var].size() > \
                    no_of_scenarios_to_consider:
                    tuple_wise_heaps[var].pop()

        return [tuple_wise_heaps[var].sum()/no_of_scenarios_to_consider\
                for var in range(self.__no_of_vars)]

    
    def __add_lcvar_constraint_to_model(
        self,
        risk_constraint: CVaRConstraint | VaRConstraint,
        cvarified_constraint: CVaRConstraint,
        no_of_scenarios: int,
        no_of_scenarios_to_consider: int
    ):
        coefficients = \
            self.__get_cvar_constraint_coefficients(
                cvarified_constraint,
                no_of_scenarios_to_consider,
                no_of_scenarios
            )
        
        gurobi_inequality = self.__get_gurobi_inequality(
            cvarified_constraint.get_inequality_sign()
        )

        self.__risk_constraints.append(risk_constraint)
        self.__risk_to_lcvar_constraint_mapping[
            risk_constraint] = \
                self.__model.addConstr(_make_constr(
                    gp.LinExpr(
                        coefficients, self.__vars),
                    gurobi_inequality,
                    cvarified_constraint.get_sum_limit()
                ))


    def __get_cvarified_constraint(
        self, constraint: VaRConstraint,
        cvar_threshold: float
    ):
        cvarified_constraint = CVaRConstraint()
        cvarified_constraint.set_percentage_of_scenarios(
            (1 - constraint.get_probability_threshold())*100)
        if constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            cvarified_constraint.set_tail_type('l')
        else:
            cvarified_constraint.set_tail_type('h')
                            
        if constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            cvarified_constraint.set_inequality_sign('>')
        else:
            cvarified_constraint.set_inequality_sign('<')
                            
        cvarified_constraint.set_attribute_name(
            constraint.get_attribute_name()
        )

        cvarified_constraint.set_sum_limit(
            cvar_threshold
        )

        return cvarified_constraint

    def __gpu_matmul(self, mat: np.ndarray, x_arr: np.ndarray) -> np.ndarray:
        """Compute mat.T @ x_arr, using GPU if available."""
        if self.__has_gpu and _TORCH_AVAILABLE:
            try:
                m = _torch.tensor(mat, dtype=_torch.float64, device='cuda')
                x = _torch.tensor(x_arr, dtype=_torch.float64, device='cuda')
                return (m.T @ x).cpu().numpy()
            except Exception:
                pass
        return mat.T @ x_arr

    def __add_z_constraint_for_scenario(self, s: int, attribute: str) -> None:
        """Add the individual Z_s constraint for scenario s to the model."""
        if s in self.__z_constrained_scenarios:
            return
        tail = self.__query.get_objective().get_tail_type()
        coeffs = [self.__scenarios[attribute][i][s] for i in range(self.__no_of_vars)]
        rhs = gp.LinExpr(coeffs, self.__vars)
        rhs.addTerms(-1.0, self.__V)
        if tail == TailType.LOWEST:
            self.__model.addConstr(self.__Zs[s] <= rhs)
        else:
            self.__model.addConstr(self.__Zs[s] >= rhs)
        self.__z_constrained_scenarios.add(s)

    def __init_z_direct_constraints(self, no_of_scenarios: int, attribute: str) -> None:
        """Add all S individual Z constraints directly to the model."""
        for s in range(no_of_scenarios):
            self.__add_z_constraint_for_scenario(s, attribute)

    def __init_z_iterative_constraints(self, no_of_scenarios: int, attribute: str) -> None:
        """Seed with one Z constraint to anchor V, then let the iterative loop add the rest.

        Without at least one Z_s constraint, V is unbounded above and the LP/ILP
        is immediately infeasible/unbounded.  A single seed scenario is the minimum
        needed to make V finite while still leaving almost all constraints for the
        iterative phase.
        """
        totals = np.array([
            sum(self.__scenarios[attribute][i][s]
                for i in range(self.__no_of_vars))
            for s in range(no_of_scenarios)
        ])
        seed = int(np.argsort(totals)[no_of_scenarios // 2])  # median scenario
        self.__add_z_constraint_for_scenario(seed, attribute)

    def __check_z_violations(self, attribute: str, tol: float = 1e-6) -> list[int]:
        """Vectorized check for violated individual Z_s constraints."""
        V_val = self.__V.x
        x_arr = np.array([v.x for v in self.__vars])
        mat = np.array([self.__scenarios[attribute][i]
                        for i in range(self.__no_of_vars)])
        lin_vals = self.__gpu_matmul(mat, x_arr)
        rhs_vals = lin_vals - V_val
        z_vals = np.array([z.x for z in self.__Zs])
        tail = self.__query.get_objective().get_tail_type()
        if tail == TailType.LOWEST:
            mask = z_vals > rhs_vals + tol
        else:
            mask = z_vals < rhs_vals - tol
        return list(np.where(mask)[0])

    def __get_violation_amounts(self, attribute: str) -> np.ndarray:
        """Compute per-scenario violation amounts for all Z constraints."""
        V_val = self.__V.x
        x_arr = np.array([v.x for v in self.__vars])
        mat = np.array([self.__scenarios[attribute][i]
                        for i in range(self.__no_of_vars)])
        lin_vals = self.__gpu_matmul(mat, x_arr)
        rhs_vals = lin_vals - V_val
        z_vals = np.array([z.x for z in self.__Zs])
        tail = self.__query.get_objective().get_tail_type()
        if tail == TailType.LOWEST:
            return np.maximum(0.0, z_vals - rhs_vals)
        else:
            return np.maximum(0.0, rhs_vals - z_vals)

    def __get_package_with_iterative_z(self):
        """Solve with iterative Z constraint addition (add p at a time)."""
        attribute = self.__query.get_objective().get_attribute_name()
        p = self.__increment_in_number_of_constraints
        while True:
            self.__metrics.start_optimizer()
            self.__model.optimize()
            self.__metrics.end_optimizer()
            status = self.__model.Status
            if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
                return None
            if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                return self.__extract_package_from_model()
            try:
                _ = self.__V.x
            except AttributeError:
                return None

            violated = self.__check_z_violations(attribute, tol=1e-6)
            unconstrained_violated = [
                s for s in violated if s not in self.__z_constrained_scenarios]
            if not unconstrained_violated:
                break

            violations = self.__get_violation_amounts(attribute)
            to_add = sorted(
                unconstrained_violated, key=lambda s: violations[s], reverse=True)[:p]
            for s in to_add:
                self.__add_z_constraint_for_scenario(s, attribute)
            print(f'Iterative Z: added {len(to_add)}, '
                  f'constrained {len(self.__z_constrained_scenarios)}/{len(self.__Zs)}')
        return self.__extract_package_from_model()

    def __rebuild_model_for_ilp_phase(
        self, no_of_scenarios: int,
        support_indices: set,
        z_scenarios: set
    ) -> None:
        """Rebuild the Gurobi model for the ILP phase of solve_lp_first."""
        self.__model = gp.Model(env=self.__gurobi_env)
        if self.__query.get_objective().is_cvar_objective():
            self.__model.Params.MIPGap = self.__approximation_bound
        max_rep = self.__get_upper_bound_for_repetitions()
        self.__vars = []
        var_idx = 0
        partition_no = 0
        for duplicates in self.__duplicate_vector:
            if max_rep is not None:
                total_ub = max_rep * self.__partition_sizes[partition_no]
            else:
                total_ub = None
            if self.__bounded_partition is not None and \
                    partition_no == self.__bounded_partition:
                bounded_ub = self.__partition_multiplicity_upper_bound
                total_ub = bounded_ub if total_ub is None else min(total_ub, bounded_ub)
            partition_no += 1
            for dup_idx in range(duplicates):
                if var_idx in support_indices:
                    if total_ub is not None:
                        ub = total_ub // duplicates
                        if dup_idx < total_ub % duplicates:
                            ub += 1
                    else:
                        ub = GRB.INFINITY
                else:
                    ub = 0
                self.__vars.append(
                    self.__model.addVar(lb=0, ub=ub, vtype=GRB.INTEGER))
                var_idx += 1

        tail = self.__query.get_objective().get_tail_type()
        self.__V = self.__model.addVar(vtype=GRB.CONTINUOUS)
        self.__Zs = []
        for _ in range(no_of_scenarios):
            if tail == TailType.LOWEST:
                self.__Zs.append(self.__model.addVar(ub=0, vtype=GRB.CONTINUOUS))
            else:
                self.__Zs.append(self.__model.addVar(lb=0, vtype=GRB.CONTINUOUS))

        any_high_tail_cvar = any(
            c.get_percentage_of_scenarios() > 50.0
            for c in self.__query.get_constraints()
            if c.is_cvar_constraint()
        )
        for constraint in self.__query.get_constraints():
            if constraint.is_package_size_constraint():
                self.__add_package_size_constraint_to_model(constraint)
            elif constraint.is_deterministic_constraint():
                self.__add_deterministic_constraint_to_model(constraint)
            elif constraint.is_expected_sum_constraint():
                self.__add_expected_sum_constraint_to_model(constraint)
            elif constraint.is_risk_constraint() and not any_high_tail_cvar:
                attr = constraint.get_attribute_name()
                sum_limit = constraint.get_sum_limit()
                coeffs = [np.median(self.__scenarios[attr][idx])
                          for idx in range(self.__no_of_vars)]
                ineq = (GRB.GREATER_EQUAL
                        if constraint.get_inequality_sign() ==
                        RelationalOperators.GREATER_THAN_OR_EQUAL_TO
                        else GRB.LESS_EQUAL)
                self.__model.addConstr(_make_constr(
                    gp.LinExpr(coeffs, self.__vars), ineq, sum_limit))

        self.__z_constrained_scenarios = set()
        attribute = self.__query.get_objective().get_attribute_name()
        for s in z_scenarios:
            self.__add_z_constraint_for_scenario(s, attribute)

        self.__add_objective_to_model(self.__query.get_objective(), no_of_scenarios)

    def __solve_with_lp_first(self):
        """LP-first: solve LP iteratively, then ILP with restricted support."""
        no_of_scenarios = self.__no_of_scenarios
        attribute = self.__query.get_objective().get_attribute_name()
        tail = self.__query.get_objective().get_tail_type()
        p = self.__increment_in_number_of_constraints

        # --- LP phase ---
        orig_lr = self.__is_linear_relaxation
        self.__is_linear_relaxation = True
        self.__model_setup(
            no_of_scenarios=no_of_scenarios,
            no_of_scenarios_to_consider=[],
            probabilistically_constrained=False
        )
        self.__is_linear_relaxation = orig_lr

        # Bound V from scenario data to prevent an unbounded LP when few Z
        # constraints are active.  With m < ceil(α·S) constraints seeded, the
        # objective grows as V·(1 − m/αS) → ∞, which causes PDHG to diverge
        # rather than quickly detecting the unbounded ray (as simplex would).
        attr_mat = np.array([self.__scenarios[attribute][i]
                             for i in range(self.__no_of_vars)])
        max_rep = self.__get_upper_bound_for_repetitions()
        max_rep = max_rep if max_rep is not None else 1
        scenario_sums = attr_mat.sum(axis=0)
        if tail == TailType.LOWEST:
            self.__V.ub = float(scenario_sums.max()) * max_rep
        else:
            self.__V.lb = float(scenario_sums.min()) * max_rep

        if self.__use_pdhg and self.__has_gpu:
            try:
                self.__model.Params.Method = 14  # PDHG (GPU)
            except Exception:
                pass
        elif self.__solve_dual_lp:
            try:
                self.__model.Params.Method = 1   # Dual simplex
            except Exception:
                pass
        else:
            try:
                self.__model.Params.Method = 0   # Primal simplex
            except Exception:
                pass

        while True:
            self.__metrics.start_optimizer()
            self.__model.optimize()
            self.__metrics.end_optimizer()
            status = self.__model.Status
            if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
                return (None, 0.0, False)
            try:
                _ = self.__V.x
            except AttributeError:
                return (None, 0.0, False)

            violated = self.__check_z_violations(attribute, tol=1e-6)
            unconstrained_violated = [
                s for s in violated if s not in self.__z_constrained_scenarios]
            if not unconstrained_violated:
                break

            violations = self.__get_violation_amounts(attribute)
            to_add = sorted(
                unconstrained_violated,
                key=lambda s: violations[s], reverse=True)[:p]
            for s in to_add:
                self.__add_z_constraint_for_scenario(s, attribute)
            print(f'LP-first LP phase: added {len(to_add)} Z constraints, '
                  f'total {len(self.__z_constrained_scenarios)}/{len(self.__Zs)}')

        lp_x_vals = np.array([v.x for v in self.__vars])
        current_support = set(int(i) for i in np.where(lp_x_vals > 1e-6)[0])
        if not current_support:
            return (None, 0.0, False)
        lp_z_constrained = set(self.__z_constrained_scenarios)

        # --- ILP phase: iterative tuple addition ---
        while True:
            self.__rebuild_model_for_ilp_phase(
                no_of_scenarios, current_support, lp_z_constrained)
            self.__metrics.start_optimizer()
            self.__model.optimize()
            self.__metrics.end_optimizer()
            status = self.__model.Status
            if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
                return (None, 0.0, False)
            try:
                _ = self.__V.x
            except AttributeError:
                return (None, 0.0, False)

            violated = self.__check_z_violations(attribute, tol=1e-6)
            if not violated:
                break

            non_support = [i for i in range(self.__no_of_vars)
                           if i not in current_support]
            if not non_support:
                break

            q = len(current_support)
            violated_arr = np.array(violated)
            mat = np.array([self.__scenarios[attribute][i]
                            for i in range(self.__no_of_vars)])
            if self.__has_gpu and _TORCH_AVAILABLE:
                try:
                    mat_gpu = _torch.tensor(
                        mat[:, violated_arr], dtype=_torch.float64, device='cuda')
                    sums = mat_gpu.sum(dim=1).cpu().numpy()
                except Exception:
                    sums = mat[:, violated_arr].sum(axis=1)
            else:
                sums = mat[:, violated_arr].sum(axis=1)

            non_support_sums = [(i, sums[i]) for i in non_support]
            if tail == TailType.LOWEST:
                non_support_sums.sort(key=lambda x: x[1], reverse=True)
            else:
                non_support_sums.sort(key=lambda x: x[1])
            for i, _ in non_support_sums[:q]:
                current_support.add(i)
            print(f'LP-first ILP phase: expanded support to {len(current_support)} tuples')

        package = self.__extract_package_from_model()
        if package is None:
            return (None, 0.0, False)
        obj_val = self.__validator.get_validation_objective_value(package)
        return (package, obj_val, False)

    def __add_constraints_to_model(
        self, no_of_scenarios: int,
        no_of_scenarios_to_consider: list[int],
        probabilistically_constrained: bool,
        cvar_thresholds: list[float],
        trivial_constraints: list[int]
    ):
        any_high_tail_cvar = any(
            c.get_percentage_of_scenarios() > 50.0
            for c in self.__query.get_constraints()
            if c.is_cvar_constraint()
        )
        stochastic_constraint_index = 0
        for constraint in self.__query.get_constraints():
            if constraint.is_package_size_constraint():
                print('Adding package size constraint to model')
                self.__add_package_size_constraint_to_model(
                    constraint
                )
            if constraint.is_deterministic_constraint():
                print('Adding deterministic constraint to model')
                self.__add_deterministic_constraint_to_model(
                    constraint
                )
            if constraint.is_expected_sum_constraint():
                threshold_override = None
                if probabilistically_constrained and \
                        stochastic_constraint_index not in trivial_constraints:
                    threshold_override = cvar_thresholds[
                        stochastic_constraint_index
                    ]
                expected_sum_model_constraint = \
                    self.__add_expected_sum_constraint_to_model(
                        constraint,
                        threshold_override=threshold_override
                    )
                if probabilistically_constrained and \
                        stochastic_constraint_index not in trivial_constraints:
                    self.__risk_to_lcvar_constraint_mapping[
                        constraint
                    ] = expected_sum_model_constraint
                stochastic_constraint_index += 1
                continue
            if constraint.is_risk_constraint():
                if probabilistically_constrained:
                    print('Considering risk constraint', stochastic_constraint_index)
                    if stochastic_constraint_index not in trivial_constraints:
                        print('Risk constraint is not trivial')
                        if constraint.is_cvar_constraint():
                            cvarified_constraint = CVaRConstraint()
                            cvarified_constraint.set_attribute_name(
                                constraint.get_attribute_name())
                            cvarified_constraint.set_percentage_of_scenarios(
                                constraint.get_percentage_of_scenarios())
                            cvarified_constraint.set_tail_type(
                                'l' if constraint.get_tail_type() == TailType.LOWEST else 'h')
                            cvarified_constraint.set_inequality_sign(
                                '>' if constraint.get_inequality_sign() == RelationalOperators.GREATER_THAN_OR_EQUAL_TO else '<')
                            cvarified_constraint.set_sum_limit(
                                cvar_thresholds[stochastic_constraint_index])
                        else:
                            print('CVaRifying VaR constraint')
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint=constraint,
                                    cvar_threshold=cvar_thresholds[
                                        stochastic_constraint_index]
                                )
                        
                        print('Adding LCVaR constraint')
                        self.__add_lcvar_constraint_to_model(
                            risk_constraint=constraint,
                            cvarified_constraint=cvarified_constraint,
                            no_of_scenarios=no_of_scenarios,
                            no_of_scenarios_to_consider=\
                                no_of_scenarios_to_consider[
                                    stochastic_constraint_index
                                ]
                        )
                else:
                    if not any_high_tail_cvar:
                        print('Probabilistically unconstrained problem')
                        print('Adding weaker constraint')
                        attribute = constraint.get_attribute_name()
                        sum_limit = constraint.get_sum_limit()
                        coefficients = []
                        for tuple_scenarios in self.__scenarios[attribute]:
                            coefficients.append(np.median(tuple_scenarios))
                        gurobi_inequality = GRB.LESS_EQUAL
                        if constraint.get_inequality_sign() == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                            gurobi_inequality = GRB.GREATER_EQUAL
                        self.__model.addConstr(_make_constr(
                            gp.LinExpr(coefficients, self.__vars), gurobi_inequality, sum_limit))
                
                stochastic_constraint_index += 1

        if not self.__check_feasibility and not self.__optimize_lcvar and \
                self.__query.get_objective().is_cvar_objective():
            attribute = self.__query.get_objective().get_attribute_name()
            self.__z_lazy_active = False
            self.__z_constrained_scenarios = set()
            if self.__iterative_constraint_addition:
                self.__z_iterative_active = True
                self.__init_z_iterative_constraints(no_of_scenarios, attribute)
            else:
                self.__z_iterative_active = False
                self.__init_z_direct_constraints(no_of_scenarios, attribute)


    def __add_objective_to_model(
        self, objective: Objective,
        no_of_scenarios: int):

        if self.__check_feasibility:
            if objective.is_cvar_objective():
                attr = objective.get_attribute_name()
                coefficients = [
                    np.average(self.__scenarios[attr][idx])
                    for idx in range(self.__no_of_vars)
                ]
                gurobi_dir = (
                    GRB.MAXIMIZE
                    if objective.get_objective_type() == ObjectiveType.MAXIMIZATION
                    else GRB.MINIMIZE
                )
                self.__model.setObjective(
                    gp.LinExpr(coefficients, self.__vars), gurobi_dir)
            else:
                raise NotImplementedError(
                    'check_feasibility=True is only supported for CVaR objectives.'
                )
            return

        if objective.is_cvar_objective():
            if self.__optimize_lcvar:
                synthetic = CVaRConstraint()
                synthetic.set_attribute_name(objective.get_attribute_name())
                synthetic.set_tail_type(
                    'l' if objective.get_tail_type() == TailType.LOWEST else 'h')
                synthetic.set_percentage_of_scenarios(
                    objective.get_percentage_of_scenarios() * 100)
                n_tail = max(1, int(np.floor(
                    objective.get_percentage_of_scenarios() * no_of_scenarios)))
                coefficients = self.__get_cvar_constraint_coefficients(
                    synthetic, n_tail, no_of_scenarios)
                gurobi_dir = (GRB.MAXIMIZE
                              if objective.get_objective_type() == ObjectiveType.MAXIMIZATION
                              else GRB.MINIMIZE)
                self.__model.setObjective(
                    gp.LinExpr(coefficients, self.__vars), gurobi_dir)
                return

            is_maximize = (objective.get_objective_type() == ObjectiveType.MAXIMIZATION)
            tail_is_lowest = (objective.get_tail_type() == TailType.LOWEST)
            if is_maximize != tail_is_lowest:
                raise NotImplementedError(
                    'Direct CVaR LP optimization is only supported for '
                    'MAXIMIZE+LOWEST and MINIMIZE+HIGHEST tail. '
                    'MINIMIZE+LOWEST and MAXIMIZE+HIGHEST are non-convex '
                    'and cannot be expressed as a single LP. '
                    'Use check_feasibility=True instead.'
                )
            coefficients = [1]
            for _ in range(len(self.__Zs)):
                coefficients.append(
                    1.0 / (objective.get_percentage_of_scenarios() * no_of_scenarios))
            if objective.get_objective_type() == ObjectiveType.MINIMIZATION:
                self.__model.setObjective(
                    gp.LinExpr(coefficients, [self.__V] + self.__Zs),
                    GRB.MINIMIZE)
            else:
                self.__model.setObjective(
                    gp.LinExpr(coefficients, [self.__V] + self.__Zs),
                    GRB.MAXIMIZE)
            return

        attr = objective.get_attribute_name()
        coefficients = []

        if objective.get_stochasticity() == \
            Stochasticity.DETERMINISTIC:
            coefficients = self.__values[attr]
        else:
            for idx in range(self.__no_of_vars):
                coefficients.append(
                    np.average(
                        self.__scenarios[attr][idx]
                    )
                )
        
        objective_type = objective.get_objective_type()

        gurobi_objective = GRB.MAXIMIZE
        if objective_type == ObjectiveType.MINIMIZATION:
            gurobi_objective = GRB.MINIMIZE

        self.__model.setObjective(
            gp.LinExpr(coefficients, self.__vars),
            gurobi_objective)
    

    def __model_setup(
        self, no_of_scenarios: int,
        no_of_scenarios_to_consider: list[int],
        probabilistically_constrained = False,
        cvar_lower_bounds = [],
        trivial_constraints = []
    ):
        
        self.__model = gp.Model(
            env=self.__gurobi_env)
        if self.__query.get_objective().is_cvar_objective():
            self.__model.Params.MIPGap = self.__approximation_bound
        if self.__check_feasibility:
            self.__model.Params.SolutionLimit = 1
            self.__model.Params.MIPFocus = 1
        print('Adding Variables to model')
        self.__add_variables_to_model(no_of_scenarios)
        print('Adding Constraints to model')
        self.__add_constraints_to_model(
            no_of_scenarios,
            no_of_scenarios_to_consider,
            probabilistically_constrained,
            cvar_lower_bounds, trivial_constraints)
        print('Adding objective to model')
        self.__add_objective_to_model(
            self.__query.get_objective(),
            no_of_scenarios)
        print('Model setup')        

    def __extract_package_from_model(self):
        """Extract solution from model variables."""
        package_dict = {}
        idx = 0
        try:
            for var in self.__vars:
                if var.x > 0:
                    package_dict[self.__ids[idx]] = var.x
                idx += 1
        except AttributeError:
            return None
        if not package_dict:
            return None
        return package_dict

    def __get_package(self):
        if (not self.__check_feasibility and not self.__optimize_lcvar
                and self.__query.get_objective().is_cvar_objective()
                and self.__z_iterative_active):
            return self.__get_package_with_iterative_z()
        self.__metrics.start_optimizer()
        self.__model.optimize()
        self.__metrics.end_optimizer()
        return self.__extract_package_from_model()


    def __get_package_with_indices(self):
        return self.__extract_package_from_model()
    
    def __get_package_tuple_partitions(self):
        package_dict = self.__extract_package_from_model()
        partition_package_dict = dict()
        for id in package_dict:
            pid = self.__partition_ids_for_each_index[id]
            if pid not in partition_package_dict:
                partition_package_dict[pid] = package_dict[id]
            else:
                partition_package_dict[pid] += package_dict[id]
        return partition_package_dict

    def __get_objective_value_among_optimization_scenarios(
        self, package_with_indices
    ) -> float:
        if package_with_indices is None:
            return 0
        objective = self.__query.get_objective()
        attr = objective.get_attribute_name()

        if objective.is_cvar_objective():
            idxs = list(package_with_indices.keys())
            mults = np.array([package_with_indices[i] for i in idxs])
            mat = np.array([
                self.__scenarios[attr][i][:self.__no_of_scenarios]
                for i in idxs
            ])
            scenario_scores = mat.T @ mults
            tail_type = objective.get_tail_type()
            if tail_type == TailType.HIGHEST:
                scenario_scores = np.sort(scenario_scores)[::-1]
            else:
                scenario_scores = np.sort(scenario_scores)
            k = max(1, int(np.floor(
                objective.get_percentage_of_scenarios() *
                self.__no_of_scenarios
            )))
            return float(np.average(scenario_scores[:k]))

        sum = 0
        for idx in package_with_indices:
            if objective.get_stochasticity() == Stochasticity.DETERMINISTIC:
                coefficient = self.__values[attr][idx]
            else:
                coefficient = np.average(self.__scenarios[attr][idx])
            sum += coefficient * package_with_indices[idx]

        return sum
    

    def __is_objective_value_relative_diff_high(
        self, package, package_with_indices,
    ):
        if self.__check_feasibility:
            return False, 0.0
        if package is None:
            return False, 0
        objective_value_optimization_scenarios =\
            self.__get_objective_value_among_optimization_scenarios(
                package_with_indices
            )
        
        objective_value_validation_scenarios =\
            self.__validator.get_validation_objective_value(
                package
            )
        
        diff = objective_value_optimization_scenarios - \
                    objective_value_validation_scenarios
        
        if diff < 0:
            diff *= -1

        rel_diff = diff / (abs(objective_value_validation_scenarios)
            + 0.00001)
        
        if rel_diff > self.__sampling_tolerance:
            return True, objective_value_validation_scenarios
        
        return False, objective_value_validation_scenarios
    

    def __is_objective_value_enough(
        self, objective_value: float,
        objective_upper_bound: float
    ):
        if self.__check_feasibility:
            return True
        objective = self.__query.get_objective()

        if objective.get_objective_type() == \
            ObjectiveType.MAXIMIZATION:
            if objective_upper_bound >= 0:
                return objective_value >= \
                    (1 - self.__approximation_bound) *\
                        objective_upper_bound
            return objective_value >= \
                (1 + self.__approximation_bound) *\
                    objective_upper_bound
        
        if objective_upper_bound >= 0:
            return objective_value <= \
                (1 + self.__approximation_bound) *\
                    objective_upper_bound
        return objective_value <= \
            (1 - self.__approximation_bound) *\
                objective_upper_bound
    

    def __get_scenario_scores_ascending(
        self, package_with_indices,
        no_of_scenarios: int,
        constraint: VaRConstraint | CVaRConstraint
    ) -> list[float]:
        attr = constraint.get_attribute_name()
        cache_key = (attr, tuple(sorted(package_with_indices.items())), no_of_scenarios)
        if cache_key in self.__opt_scenario_scores_cache:
            return self.__opt_scenario_scores_cache[cache_key]
        idxs = list(package_with_indices.keys())
        mults = np.array([package_with_indices[i] for i in idxs])
        mat = np.array([self.__scenarios[attr][i][:no_of_scenarios] for i in idxs])
        scenario_scores = np.sort(mat.T @ mults).tolist()
        self.__opt_scenario_scores_cache[cache_key] = scenario_scores
        return scenario_scores


    def __is_var_constraint_satisfied(
        self, var_constraint: VaRConstraint,
        var_validation: float
    ) -> bool:
        if var_constraint.get_inequality_sign() ==\
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            return var_validation >=\
                var_constraint.get_sum_limit()
        
        return var_validation <=\
            var_constraint.get_sum_limit()
    

    def __is_cvar_constraint_satisfied(
        self, cvar_constraint: CVaRConstraint,
        cvar_validation: float
    ) -> bool:
        if cvar_constraint.get_inequality_sign() == \
            RelationalOperators.LESS_THAN_OR_EQUAL_TO:
            return (cvar_validation <=\
                        cvar_constraint.get_sum_limit())
        
        return (cvar_validation >=\
                    cvar_constraint.get_sum_limit())

    
    def __l_inf(self, l1: list[int | float],
                l2: list[int | float]) -> float:
        if len(l1) != len(l2):
            raise Exception
        max_abs_diff = 0

        for ind in range(len(l1)):
            diff = l1[ind] - l2[ind]
            if diff < 0:
                diff *= -1
            if diff > max_abs_diff:
                max_abs_diff = diff
        
        return max_abs_diff


    def __repair_lower_bound_anchors(
        self,
        no_of_scenarios: int,
        cvar_upper_bounds: list[float | int],
        cvar_lower_bounds: list[float | int],
        trivial_constraints: list[int],
        no_of_scenarios_to_consider: list[int],
        can_add_scenarios: bool
    ) -> tuple[list[float | int], bool, bool, object, float | None]:
        cvar_lower_bounds = list(cvar_lower_bounds)
        if len(cvar_lower_bounds) == 0:
            return cvar_lower_bounds, False, False, None, None

        print('[LowerRepair] Starting | upper bounds:', cvar_upper_bounds,
              '| lower bounds:', cvar_lower_bounds,
              '| scenarios:', no_of_scenarios_to_consider)
        is_model_setup = False
        best_feasible_package = None
        best_feasible_objective = None
        repair_iterations = 0

        while True:
            print('[LowerRepair] Trying lower bounds:', cvar_lower_bounds)
            if not is_model_setup:
                self.__model_setup(
                    no_of_scenarios=no_of_scenarios,
                    no_of_scenarios_to_consider=no_of_scenarios_to_consider,
                    probabilistically_constrained=True,
                    cvar_lower_bounds=cvar_lower_bounds,
                    trivial_constraints=trivial_constraints
                )
                is_model_setup = True
            else:
                stochastic_constraint_index = 0
                for constraint in self.__get_searchable_constraints():
                    if stochastic_constraint_index not in trivial_constraints:
                        self.__risk_to_lcvar_constraint_mapping[
                            constraint].rhs = cvar_lower_bounds[
                                stochastic_constraint_index]
                    stochastic_constraint_index += 1

            package = self.__get_package()
            print('[LowerRepair] Package:', package)
            if package is None:
                print('[LowerRepair] No package found; keeping current'
                      ' lower bounds as heuristic anchors.')
                return cvar_lower_bounds, is_model_setup, False, best_feasible_package, best_feasible_objective

            package_with_indices = self.__get_package_with_indices()
            unacceptable_diff, _ = \
                self.__is_objective_value_relative_diff_high(
                    package, package_with_indices,
                    no_of_scenarios
                )
            if unacceptable_diff and can_add_scenarios:
                print('[LowerRepair] Needs more scenarios (obj diff)')
                return cvar_lower_bounds, is_model_setup, True, best_feasible_package, best_feasible_objective

            violated_constraints = []
            stochastic_constraint_index = 0
            for constraint in self.__get_searchable_constraints():
                is_trivial = stochastic_constraint_index in trivial_constraints
                if is_trivial:
                    stochastic_constraint_index += 1
                    continue

                if constraint.is_expected_sum_constraint():
                    satisfied = \
                        self.__validator.get_expected_sum_constraint_feasibility(
                            package, constraint
                        )
                elif constraint.is_cvar_constraint():
                    cvar_validation = \
                        self.__validator.get_cvar_among_validation_scenarios(
                            package, constraint
                        )
                    satisfied = self.__is_cvar_constraint_satisfied(
                        constraint, cvar_validation
                    )
                else:
                    var_validation = \
                        self.__validator.get_var_among_validation_scenarios(
                            package, constraint
                        )
                    satisfied = self.__is_var_constraint_satisfied(
                        constraint, var_validation
                    )

                if not satisfied:
                    violated_constraints.append(stochastic_constraint_index)
                stochastic_constraint_index += 1

            if len(violated_constraints) == 0:
                print('[LowerRepair] Lower bounds validated.')
                if best_feasible_package is None:
                    best_feasible_package = package
                    best_feasible_objective = self.__validator.get_validation_objective_value(package)
                else:
                    cand_obj = self.__validator.get_validation_objective_value(package)
                    if (self.__query.get_objective().get_objective_type() == ObjectiveType.MAXIMIZATION and cand_obj > best_feasible_objective) or \
                       (self.__query.get_objective().get_objective_type() != ObjectiveType.MAXIMIZATION and cand_obj < best_feasible_objective):
                        best_feasible_package = package
                        best_feasible_objective = cand_obj
                return cvar_lower_bounds, is_model_setup, False, best_feasible_package, best_feasible_objective

            print('[LowerRepair] Violated constraint indices:',
                  violated_constraints)
            for violated_index in violated_constraints:
                diff = abs(
                    cvar_upper_bounds[violated_index] -
                    cvar_lower_bounds[violated_index]
                )
                if cvar_lower_bounds[violated_index] <= \
                        cvar_upper_bounds[violated_index]:
                    cvar_lower_bounds[violated_index] -= diff
                else:
                    cvar_lower_bounds[violated_index] += diff
            print('[LowerRepair] Updated lower bounds:',
                  cvar_lower_bounds)
            repair_iterations += 1
            if repair_iterations >= 10:
                print('[LowerRepair] Max iterations reached; requesting more scenarios.')
                return cvar_lower_bounds, is_model_setup, True, best_feasible_package, best_feasible_objective

    def __cvar_threshold_search(
        self, no_of_scenarios: int,
        cvar_upper_bounds: list[float | int],
        cvar_lower_bounds: list[float | int],
        trivial_constraints: list[int],
        objective_upper_bound: float,
        no_of_scenarios_to_consider: list[int],
        is_model_setup: bool,
        can_add_scenarios: bool
    ) -> CVaRificationSearchResults:
        cvar_upper_bounds = list(cvar_upper_bounds)
        cvar_lower_bounds = list(cvar_lower_bounds)
        best_feasible_package = None
        best_feasible_objective = None
        found_non_none_infeasible = False
        if not is_model_setup and \
                self.__l_inf(cvar_upper_bounds, cvar_lower_bounds) < \
                self.__bisection_threshold:
            cvar_initial_thresholds = [
                (u + l) / 2.0
                for u, l in zip(cvar_upper_bounds, cvar_lower_bounds)
            ]
            self.__model_setup(
                no_of_scenarios=no_of_scenarios,
                no_of_scenarios_to_consider=no_of_scenarios_to_consider,
                probabilistically_constrained=True,
                cvar_lower_bounds=cvar_initial_thresholds,
                trivial_constraints=trivial_constraints
            )
            is_model_setup = True
        while self.__l_inf(
            cvar_upper_bounds, cvar_lower_bounds) >= \
            self.__bisection_threshold:

            cvar_mid_thresholds = []
            for ind in range(len(cvar_lower_bounds)):
                cvar_mid_thresholds.append(
                    (cvar_lower_bounds[ind] + \
                    cvar_upper_bounds[ind]) / 2.0
                )

            print('Bisecting for thresholds')
            print('CVaR upper bounds:', cvar_upper_bounds)
            print('CVaR lower bounds:', cvar_lower_bounds)
            print('CVaR mid thresholds:', cvar_mid_thresholds)

            if not is_model_setup:
                self.__model_setup(
                    no_of_scenarios=no_of_scenarios,
                    no_of_scenarios_to_consider=\
                        no_of_scenarios_to_consider,
                    probabilistically_constrained=True,
                    cvar_lower_bounds=cvar_mid_thresholds,
                    trivial_constraints=trivial_constraints
                )
                is_model_setup = True
            
            else:
                stochastic_constraint_index = 0
                for constraint in self.__get_searchable_constraints():
                    if stochastic_constraint_index not in \
                        trivial_constraints:
                        self.__risk_to_lcvar_constraint_mapping[
                            constraint].rhs = cvar_mid_thresholds[
                                stochastic_constraint_index]
                    stochastic_constraint_index += 1
            
            package = self.__get_package()
            print('Package:', package)
            if package is None:
                cvar_lower_bounds = cvar_mid_thresholds
                continue
            
            package_with_indices = \
                package
            
            unacceptable_diff, validation_objective_value = \
                self.__is_objective_value_relative_diff_high(
                    package, package_with_indices
                )
            
            if unacceptable_diff and can_add_scenarios:
                print('Unacceptable diff for objective')
                result = CVaRificationSearchResults()
                result.set_needs_more_scenarios(True)
                return result
            
            all_constraints_satisfied = True
            all_constraints_violated = True
            stochastic_constraint_index = 0
            
            for constraint in self.__get_searchable_constraints():
                is_trivial = stochastic_constraint_index in trivial_constraints
                if constraint.is_expected_sum_constraint():
                    expected_sum_satisfied = \
                        self.__validator.get_expected_sum_constraint_feasibility(
                            package, constraint
                        )
                    if expected_sum_satisfied:
                        print('Expected sum constraint satisfied')
                        all_constraints_violated = False
                        if not is_trivial:
                            cvar_lower_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                    else:
                        print('Expected sum constraint not satisfied')
                        all_constraints_satisfied = False
                        if not is_trivial:
                            cvar_upper_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                    stochastic_constraint_index += 1
                    continue

                if not is_trivial:
                    if constraint.is_cvar_constraint():
                        cvar_validation = \
                            self.__validator.get_cvar_among_validation_scenarios(
                                package, constraint
                            )
                        
                        if self.__is_cvar_constraint_satisfied(
                            constraint, cvar_validation
                        ):
                            print('CVaR constraint satisfied')
                            all_constraints_violated = False
                            cvar_lower_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                        else:
                            print('CVaR constraint not satisfied')
                            all_constraints_satisfied = False
                            cvar_upper_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                
                    if constraint.is_var_constraint():
                        var_validation = \
                            self.__validator.get_var_among_validation_scenarios(
                                package, constraint
                            )
                        if self.__is_var_constraint_satisfied(
                            constraint, var_validation
                        ):
                            print('VaR constraint satisfied')
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint,
                                    constraint.get_sum_limit()
                                )
                            cvarified_cvar_validation = \
                                self.__validator.get_cvar_among_validation_scenarios(
                                    package, cvarified_constraint
                                )
                            if self.__is_cvar_constraint_satisfied(
                                cvarified_constraint,
                                cvarified_cvar_validation
                            ):
                                all_constraints_violated = False
                            cvar_lower_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                        else:
                            print('VaR constraint not satisfied')
                            all_constraints_satisfied = False
                            cvar_upper_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                stochastic_constraint_index += 1

            if all_constraints_satisfied:
                if self.__is_objective_value_enough(
                    validation_objective_value, objective_upper_bound
                ):
                    result = CVaRificationSearchResults()
                    result.set_found_appropriate_package(True)
                    result.set_package(package)
                    result.set_objective_value(validation_objective_value)
                    return result
                else:
                    if best_feasible_package is None or \
                        (self.__query.get_objective().get_objective_type() == \
                            ObjectiveType.MAXIMIZATION and \
                            validation_objective_value > best_feasible_objective) or \
                        (self.__query.get_objective().get_objective_type() != \
                            ObjectiveType.MAXIMIZATION and \
                            validation_objective_value < best_feasible_objective):
                        best_feasible_package = package
                        best_feasible_objective = validation_objective_value
            else:
                # Package is not None but constraints are not satisfied
                if package is not None:
                    found_non_none_infeasible = True

            if all_constraints_violated:
                if self.__query.get_objective().get_objective_type() ==\
                    ObjectiveType.MAXIMIZATION:
                    if validation_objective_value < objective_upper_bound:
                        objective_upper_bound = validation_objective_value
                else:
                    if validation_objective_value > objective_upper_bound:
                        objective_upper_bound = validation_objective_value

        if best_feasible_package is not None and \
            self.__is_objective_value_enough(
                best_feasible_objective, objective_upper_bound):
            result = CVaRificationSearchResults()
            result.set_found_appropriate_package(True)
            result.set_package(best_feasible_package)
            result.set_objective_value(best_feasible_objective)
            return result

        result = CVaRificationSearchResults()
        result.set_objective_upper_bound(
            objective_upper_bound)
        result.set_cvar_thresholds(
            cvar_lower_bounds)
        if best_feasible_package is not None:
            result.set_best_feasible_package(best_feasible_package)
            result.set_best_feasible_objective(best_feasible_objective)
        result.set_found_non_none_infeasible(found_non_none_infeasible)
        return result
            

    def __cvar_coefficient_search(
        self, no_of_scenarios: int,
        cvar_thresholds: list[float | int],
        trivial_constraints: list[int],
        objective_upper_bound: float,
        min_no_of_scenarios_to_consider: list[int],
        max_no_of_scenarios_to_consider: list[int],
        is_model_setup: bool,
        can_add_scenarios: bool
    ) -> CVaRificationSearchResults:
        max_no_of_scenarios_to_consider = list(max_no_of_scenarios_to_consider)
        best_feasible_package = None
        best_feasible_objective = None
        found_non_none_infeasible = False
        while self.__l_inf(
            min_no_of_scenarios_to_consider,
            max_no_of_scenarios_to_consider
        ) > 1:
            
            mid_no_of_scenarios_to_consider = []
            for ind in range(len(min_no_of_scenarios_to_consider)):
                mid_no_of_scenarios_to_consider.append(
                    (min_no_of_scenarios_to_consider[ind] +\
                     max_no_of_scenarios_to_consider[ind]) // 2
                )
            
            print('Solving by considering', mid_no_of_scenarios_to_consider,
                  'for each constraint')
            
            if not is_model_setup:
                print('Setting up probabilistically constrained model')
                self.__model_setup(
                    no_of_scenarios=no_of_scenarios,
                    no_of_scenarios_to_consider=\
                        mid_no_of_scenarios_to_consider,
                    probabilistically_constrained=True,
                    cvar_lower_bounds=cvar_thresholds,
                    trivial_constraints=trivial_constraints
                )
                is_model_setup = True
            else:
                print('Using existing model')
                stochastic_constraint_index = 0
                for constraint in self.__get_searchable_constraints():
                    if constraint.is_risk_constraint() and \
                            stochastic_constraint_index not in trivial_constraints:
                            print('Risk constraint', stochastic_constraint_index,
                                  'is not trivial')
                            print('Getting constraints with',
                                  mid_no_of_scenarios_to_consider[stochastic_constraint_index],
                                  'scenarios from each tuple')
                            if constraint.is_cvar_constraint():
                                cvarified_constraint = CVaRConstraint()
                                cvarified_constraint.set_attribute_name(
                                    constraint.get_attribute_name())
                                cvarified_constraint.set_percentage_of_scenarios(
                                    constraint.get_percentage_of_scenarios())
                                cvarified_constraint.set_tail_type(
                                    'l' if constraint.get_tail_type() == TailType.LOWEST else 'h')
                                cvarified_constraint.set_inequality_sign(
                                    '>' if constraint.get_inequality_sign() == RelationalOperators.GREATER_THAN_OR_EQUAL_TO else '<')
                                cvarified_constraint.set_sum_limit(
                                    cvar_thresholds[stochastic_constraint_index])
                            else:
                                cvarified_constraint = \
                                    self.__get_cvarified_constraint(
                                        constraint, cvar_thresholds[
                                            stochastic_constraint_index])
                            coefficients = \
                                self.__get_cvar_constraint_coefficients(
                                    cvarified_constraint,
                                    mid_no_of_scenarios_to_consider[
                                        stochastic_constraint_index],
                                    no_of_scenarios)
                        
                            for _ in range(len(self.__vars)):
                                self.__model.chgCoeff(
                                    self.__risk_to_lcvar_constraint_mapping[
                                        constraint],
                                    self.__vars[_], coefficients[_]
                                )
                            print('Changing constraint coefficients')                    
                    stochastic_constraint_index += 1
            
            print('Getting package with new coefficients')
            package = self.__get_package()
            print('Obtained package:', package)
            if package is None:
                min_no_of_scenarios_to_consider = \
                    mid_no_of_scenarios_to_consider
                continue

            package_with_indices = \
                package
            
            unacceptable_diff, validation_objective_value = \
                self.__is_objective_value_relative_diff_high(
                    package, package_with_indices
                )
            
            if unacceptable_diff and can_add_scenarios:
                result = CVaRificationSearchResults()
                result.set_needs_more_scenarios(True)
                print('Diff between opt and val scenario'
                      'values is high.')
                print('Asking to add more scenarios.')
                return result
            
            all_constraints_satisfied = True
            all_constraints_violated = True
            stochastic_constraint_index = 0

            for constraint in self.__get_searchable_constraints():
                is_trivial = stochastic_constraint_index in trivial_constraints
                if constraint.is_expected_sum_constraint():
                    expected_sum_satisfied = \
                        self.__validator.get_expected_sum_constraint_feasibility(
                            package, constraint
                        )
                    if expected_sum_satisfied:
                        print('Expected sum constraint satisfied')
                        all_constraints_violated = False
                    else:
                        print('Expected sum constraint not satisfied')
                        all_constraints_satisfied = False
                    stochastic_constraint_index += 1
                    continue

                if not is_trivial:
                    if constraint.is_cvar_constraint():
                        print('Checking CVaR constraint satisfaction')
                        cvar_validation = \
                            self.__validator.get_cvar_among_validation_scenarios(
                                package, constraint
                            )

                        if self.__is_cvar_constraint_satisfied(
                            constraint, cvar_validation
                        ):
                            print('CVaR constraint is satisfied')
                            all_constraints_violated = False
                            min_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index]
                            print('Relaxing risk constraint', stochastic_constraint_index,
                                  'by updating the min. scenario bound to',
                                  min_no_of_scenarios_to_consider[stochastic_constraint_index])
                        else:
                            print('CVaR constraint is not satisfied')
                            all_constraints_satisfied = False
                            max_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index] - 1
                            print('Tightening risk constraint', stochastic_constraint_index,
                                  'by updating the max. scenario bound to',
                                  max_no_of_scenarios_to_consider[stochastic_constraint_index])

                    if constraint.is_var_constraint():
                        print('Checking VaR constraint satisfaction')
                        var_validation = \
                            self.__validator.get_var_among_validation_scenarios(
                                package, constraint
                            )

                        if self.__is_var_constraint_satisfied(
                            constraint, var_validation
                        ):
                            print('VaR constraint satisfied')
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint,
                                    constraint.get_sum_limit()
                                )
                            cvarified_cvar_validation = \
                                self.__validator.get_cvar_among_validation_scenarios(
                                    package, cvarified_constraint
                                )
                            if self.__is_cvar_constraint_satisfied(
                                cvarified_constraint,
                                cvarified_cvar_validation
                            ):
                                all_constraints_violated = False
                            min_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index]
                            print('Updating min no. of scenarios to consider for risk'
                                  'constraint', stochastic_constraint_index, 'to',
                                  min_no_of_scenarios_to_consider[stochastic_constraint_index])
                        else:
                            print('VaR constraint not satisfied')
                            all_constraints_satisfied = False
                            max_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index] - 1
                            print('Updating max no. of scenarios to consider for risk',
                                  'constraint', stochastic_constraint_index, 'to',
                                  max_no_of_scenarios_to_consider[stochastic_constraint_index])
                stochastic_constraint_index += 1

            if all_constraints_satisfied:
                print('All constraints are satisfied')
                if self.__is_objective_value_enough(
                    validation_objective_value, objective_upper_bound
                ):
                    print('Objective value is enough')
                    result = CVaRificationSearchResults()
                    result.set_found_appropriate_package(True)
                    result.set_package(package)
                    result.set_objective_value(
                        validation_objective_value)
                    return result
                else:
                    print('Objective value is insufficient')
                    if best_feasible_package is None or \
                        (self.__query.get_objective().get_objective_type() == \
                            ObjectiveType.MAXIMIZATION and \
                            validation_objective_value > best_feasible_objective) or \
                        (self.__query.get_objective().get_objective_type() != \
                            ObjectiveType.MAXIMIZATION and \
                            validation_objective_value < best_feasible_objective):
                        best_feasible_package = package
                        best_feasible_objective = validation_objective_value
            else:
                # Package is not None but constraints are not satisfied
                if package is not None:
                    found_non_none_infeasible = True

            if all_constraints_violated:
                if self.__query.get_objective().get_objective_type() ==\
                    ObjectiveType.MAXIMIZATION:
                    if validation_objective_value < objective_upper_bound:
                        objective_upper_bound = validation_objective_value
                else:
                    if validation_objective_value > objective_upper_bound:
                        objective_upper_bound = validation_objective_value

        if best_feasible_package is not None and \
            self.__is_objective_value_enough(
                best_feasible_objective, objective_upper_bound):
            print('Best feasible package from coefficient search passes updated bound.')
            result = CVaRificationSearchResults()
            result.set_found_appropriate_package(True)
            result.set_package(best_feasible_package)
            result.set_objective_value(best_feasible_objective)
            return result

        print('Could not find a satisfactory package with the coefficient search.')
        result = CVaRificationSearchResults()
        result.set_objective_upper_bound(objective_upper_bound)
        print('Updating the min no of scenarios to consider to',
              min_no_of_scenarios_to_consider)
        result.set_scenarios_to_consider(min_no_of_scenarios_to_consider)
        if best_feasible_package is not None:
            result.set_best_feasible_package(best_feasible_package)
            result.set_best_feasible_objective(best_feasible_objective)
        result.set_found_non_none_infeasible(found_non_none_infeasible)
        return result
            
    
    def __get_linearized_cvar_among_optimization_scenarios(
        self, probabilistically_unconstrained_package,
        cvar_constraint, no_of_scenarios
    ):
        no_of_scenarios_to_consider = \
            self.__get_no_of_scenarios_to_consider(
                cvar_constraint, no_of_scenarios
            )
        
        coefficients = self.__get_cvar_constraint_coefficients(
            cvar_constraint,
            no_of_scenarios_to_consider,
            no_of_scenarios
        )

        linearized_cvar = 0
        idx = 0
        for id in self.__ids:
            if id in probabilistically_unconstrained_package:
                linearized_cvar += \
                    coefficients[idx] *\
                        probabilistically_unconstrained_package[id]
            idx += 1
        return linearized_cvar
    
    def __get_expected_sum_among_optimization_scenarios(
        self, probabilistically_unconstrained_package,
        attribute
    ):
        expected_sum = 0
        for id in probabilistically_unconstrained_package:
            expected_sum += np.mean(
                self.__scenarios[attribute][id])*\
                    probabilistically_unconstrained_package[id]
        return expected_sum
    

    def __get_bounds_for_risk_constraints(
        self, no_of_scenarios,
        probabilistically_unconstrained_package
    ):
        stochastic_constraint_index = 0
        cvar_lower_bounds = []
        cvar_upper_bounds = []
        min_no_of_scenarios_to_consider = []
        max_no_of_scenarios_to_consider = []
        trivial_constraints = []
        for constraint in self.__get_searchable_constraints():
            is_satisfied = False
            if constraint.is_expected_sum_constraint():
                if self.__validator.get_expected_sum_constraint_feasibility(
                    probabilistically_unconstrained_package,
                    constraint
                ):
                    is_satisfied = True
            elif constraint.is_var_constraint():
                if self.__validator.get_var_constraint_feasibility(
                    probabilistically_unconstrained_package,
                    constraint):
                    is_satisfied = True
            elif constraint.is_cvar_constraint():
                if self.__validator.get_cvar_constraint_feasibility(
                    probabilistically_unconstrained_package,
                    constraint):
                    is_satisfied = True
            if is_satisfied:
                trivial_constraints.append(stochastic_constraint_index)
                cvar_lower_bounds.append(constraint.get_sum_limit())
                cvar_upper_bounds.append(constraint.get_sum_limit())
                if constraint.is_expected_sum_constraint():
                    no_of_scenarios_to_consider = 1
                elif constraint.is_cvar_constraint():
                    no_of_scenarios_to_consider =\
                        max(1, int(np.floor(constraint.get_percentage_of_scenarios()\
                                 *no_of_scenarios/100.0)))
                else:
                    no_of_scenarios_to_consider =\
                        max(1, int(np.ceil((1 - constraint.get_probability_threshold())\
                                 *no_of_scenarios)))
                min_no_of_scenarios_to_consider.append(
                    no_of_scenarios_to_consider
                )
                max_no_of_scenarios_to_consider.append(
                    no_of_scenarios_to_consider
                )
                stochastic_constraint_index += 1
                continue
            
            if constraint.is_expected_sum_constraint():
                expected_sum = \
                    self.__get_expected_sum_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        constraint.get_attribute_name()
                    )
                
                cvar_upper_bounds.append(expected_sum)
                cvar_lower_bounds.append(constraint.get_sum_limit())
                
                min_no_of_scenarios_to_consider.append(1)
                max_no_of_scenarios_to_consider.append(1)
                
                stochastic_constraint_index += 1
                continue

            if constraint.is_cvar_constraint():
                cvar_threshold = \
                    self.__get_linearized_cvar_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        constraint, no_of_scenarios
                    )
                
                cvar_upper_bounds.append(cvar_threshold)
                
                expected_sum = \
                    self.__get_expected_sum_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        constraint.get_attribute_name())
                
                cvar_lower_bounds.append(expected_sum)
                
                no_of_scenarios_to_consider =\
                    max(1, int(np.floor(
                        constraint.get_percentage_of_scenarios()\
                         *no_of_scenarios/100.0)))
                min_no_of_scenarios_to_consider.append(
                    no_of_scenarios_to_consider
                )
                max_no_of_scenarios_to_consider.append(
                    no_of_scenarios
                )
            else:
                cvarified_constraint = \
                    self.__get_cvarified_constraint(
                        constraint, constraint.get_sum_limit())

                cvar_threshold = \
                    self.__get_linearized_cvar_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        cvarified_constraint, no_of_scenarios
                    )
                cvar_upper_bounds.append(cvar_threshold)
                
                cvar_lower_bounds.append(
                    self.__get_expected_sum_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        constraint.get_attribute_name()))
                
                no_of_scenarios_to_consider =\
                    max(1, int(np.floor(
                        cvarified_constraint.get_percentage_of_scenarios()\
                         *no_of_scenarios/100.0)))
                
                min_no_of_scenarios_to_consider.append(
                    no_of_scenarios_to_consider
                )
                max_no_of_scenarios_to_consider.append(
                    no_of_scenarios
                )
            stochastic_constraint_index += 1

        for idx in range(len(cvar_lower_bounds)):
            if cvar_lower_bounds[idx] > cvar_upper_bounds[idx]:
                cvar_upper_bounds[idx] -= (
                    cvar_lower_bounds[idx] - cvar_upper_bounds[idx]
                )
            else:
                cvar_upper_bounds[idx] -= (
                    cvar_lower_bounds[idx] - cvar_upper_bounds[idx]
                )
        return cvar_upper_bounds, cvar_lower_bounds,\
            max_no_of_scenarios_to_consider,\
            min_no_of_scenarios_to_consider,\
            trivial_constraints
    

    def solve(self, can_add_scenarios = False):
        self.__metrics.start_execution()
        if self.__solve_lp_first and \
                self.__query.get_objective().is_cvar_objective():
            result = self.__solve_with_lp_first()
            self.__metrics.end_execution(
                result[1] if result[0] is not None else 0.0,
                self.__no_of_scenarios
            )
            return result
        no_of_scenarios = self.__no_of_scenarios

        print('Setting up model')
        self.__model_setup(
            no_of_scenarios=no_of_scenarios,
            no_of_scenarios_to_consider=[],
            probabilistically_constrained=False
        )

        probabilistically_unconstrained_package = \
            self.__get_package()
        if probabilistically_unconstrained_package is not None:
            print('Probabilistically unconstrained package:',
                self.__get_package_tuple_partitions())
        else:
            print('Probabilistically unconstrained package: None')
        if probabilistically_unconstrained_package is None:
            print('Probabilistically unconstrained'
                  'problem is infeasible')
            self.__metrics.end_execution(0, 0)
            return (None, 0.0, False)
        
        probabilistically_unconstrained_package_with_indices = \
            probabilistically_unconstrained_package

        unacceptable_diff, validation_objective_value = \
            self.__is_objective_value_relative_diff_high(
                probabilistically_unconstrained_package,
                probabilistically_unconstrained_package_with_indices
            )
            
        objective_upper_bound = validation_objective_value

        if can_add_scenarios and unacceptable_diff:
            return (None, 0.0, True)

        print('Objective value upper bound:',
              objective_upper_bound)

        if self.__validator.is_package_validation_feasible(
            probabilistically_unconstrained_package):
            print('Probabilistically unconstrained package'
                  'is validation feasible')
            self.__metrics.end_execution(
                objective_upper_bound, 0)
            return (probabilistically_unconstrained_package,
                    objective_upper_bound, False)

        cvar_upper_bounds, cvar_lower_bounds,\
        max_no_of_scenarios_to_consider,\
        min_no_of_scenarios_to_consider,\
        trivial_constraints = \
            self.__get_bounds_for_risk_constraints(
                no_of_scenarios,
                probabilistically_unconstrained_package,
            )

        self.__preprocess_lcvar_prefix_sums()

        is_model_setup = False
        global_best_feasible_package = None
        global_best_feasible_objective = None

        print('CVaR upper bounds:', cvar_upper_bounds)
        print('CVaR lower bounds:', cvar_lower_bounds)
        print('Min No of Scenarios to consider:',
              min_no_of_scenarios_to_consider)
        print('Max no of scenarios to consider:',
              max_no_of_scenarios_to_consider)
        print('Trivial constraints:', trivial_constraints)

        cvar_lower_bounds, is_model_setup, needs_more_scenarios, repair_feasible_package, repair_feasible_objective = \
            self.__repair_lower_bound_anchors(
                no_of_scenarios=no_of_scenarios,
                cvar_upper_bounds=cvar_upper_bounds,
                cvar_lower_bounds=cvar_lower_bounds,
                trivial_constraints=trivial_constraints,
                no_of_scenarios_to_consider=\
                    max_no_of_scenarios_to_consider,
                can_add_scenarios=can_add_scenarios
            )

        if repair_feasible_package is not None:
            if global_best_feasible_package is None or \
                (self.__query.get_objective().get_objective_type() == ObjectiveType.MAXIMIZATION and repair_feasible_objective > global_best_feasible_objective) or \
                (self.__query.get_objective().get_objective_type() != ObjectiveType.MAXIMIZATION and repair_feasible_objective < global_best_feasible_objective):
                global_best_feasible_package = repair_feasible_package
                global_best_feasible_objective = repair_feasible_objective

        if can_add_scenarios and needs_more_scenarios:
            return (None, 0.0, True)

        print('CVaR lower bounds after repair:', cvar_lower_bounds)

        init_diff = max(
            self.__l_inf(cvar_upper_bounds, cvar_lower_bounds),
            1e-9
        )

        while self.__l_inf(cvar_upper_bounds, cvar_lower_bounds)\
            >= self.__bisection_threshold*init_diff or\
                self.__l_inf(min_no_of_scenarios_to_consider,
                             max_no_of_scenarios_to_consider) >= 1:
                
            print('CVaR upper bounds:', cvar_upper_bounds)
            print('CVaR lower bounds:', cvar_lower_bounds)
            print('Min No of Scenarios to consider:',
                min_no_of_scenarios_to_consider)
            print('Max no of scenarios to consider:',
                max_no_of_scenarios_to_consider)
            print('Trivial constraints:', trivial_constraints)

            print('Moving on to threshold search')
            threshold_search_result = \
                self.__cvar_threshold_search(
                    no_of_scenarios,
                    cvar_upper_bounds,
                    cvar_lower_bounds,
                    trivial_constraints,
                    objective_upper_bound,
                    min_no_of_scenarios_to_consider,
                    is_model_setup,
                    can_add_scenarios
                )

            is_model_setup = True

            if can_add_scenarios and threshold_search_result.\
                get_needs_more_scenarios():
                return (None, 0.0, True)

            if threshold_search_result.\
                get_found_appropriate_package():
                self.__metrics.end_execution(
                    threshold_search_result.\
                        get_objective_value(),
                    no_of_scenarios
                )
                return (
                    threshold_search_result.get_package(),
                    threshold_search_result.get_objective_value(),
                    False
                )
            cvar_upper_bounds = \
                threshold_search_result.get_cvar_thresholds()
            objective_upper_bound = \
                threshold_search_result.get_objective_upper_bound()

            if threshold_search_result.get_best_feasible_package() is not None:
                cand_pkg = threshold_search_result.get_best_feasible_package()
                cand_obj = threshold_search_result.get_best_feasible_objective()
                if global_best_feasible_package is None or \
                    (self.__query.get_objective().get_objective_type() == \
                        ObjectiveType.MAXIMIZATION and cand_obj > global_best_feasible_objective) or \
                    (self.__query.get_objective().get_objective_type() != \
                        ObjectiveType.MAXIMIZATION and cand_obj < global_best_feasible_objective):
                    global_best_feasible_package = cand_pkg
                    global_best_feasible_objective = cand_obj

            # Check if global best is within approximation bound of objective upper bound
            if global_best_feasible_package is not None and \
                    self.__is_objective_value_enough(global_best_feasible_objective, objective_upper_bound):
                print('Best feasible package within approximation bound; returning.')
                self.__metrics.end_execution(global_best_feasible_objective, no_of_scenarios)
                return (global_best_feasible_package, global_best_feasible_objective, False)

            coefficient_search_result = \
                self.__cvar_coefficient_search(
                    no_of_scenarios,
                    cvar_upper_bounds,
                    trivial_constraints,
                    objective_upper_bound,
                    min_no_of_scenarios_to_consider,
                    max_no_of_scenarios_to_consider,
                    is_model_setup,
                    can_add_scenarios,
                )

            if coefficient_search_result.\
                get_needs_more_scenarios():
                print('Needs more scenarios')
                return (None, 0.0, True)
            if coefficient_search_result.\
                get_found_appropriate_package():
                print('Coefficient search found the appropriate package')
                self.__metrics.end_execution(
                    coefficient_search_result.\
                        get_objective_value(),
                    no_of_scenarios
                )
                return (
                    coefficient_search_result.get_package(),
                    coefficient_search_result.get_objective_value(),
                    False
                )

            min_no_of_scenarios_to_consider = \
                coefficient_search_result.get_scenarios_to_consider()
            objective_upper_bound = \
                coefficient_search_result.get_objective_upper_bound()

            if coefficient_search_result.get_best_feasible_package() is not None:
                cand_pkg = coefficient_search_result.get_best_feasible_package()
                cand_obj = coefficient_search_result.get_best_feasible_objective()
                if global_best_feasible_package is None or \
                    (self.__query.get_objective().get_objective_type() == \
                        ObjectiveType.MAXIMIZATION and cand_obj > global_best_feasible_objective) or \
                    (self.__query.get_objective().get_objective_type() != \
                        ObjectiveType.MAXIMIZATION and cand_obj < global_best_feasible_objective):
                    global_best_feasible_package = cand_pkg
                    global_best_feasible_objective = cand_obj

            # Check if global best is within approximation bound of objective upper bound
            if global_best_feasible_package is not None and \
                    self.__is_objective_value_enough(global_best_feasible_objective, objective_upper_bound):
                print('Best feasible package within approximation bound; returning.')
                self.__metrics.end_execution(global_best_feasible_objective, no_of_scenarios)
                return (global_best_feasible_package, global_best_feasible_objective, False)

            # Convergence check: if no non-None infeasible packages found in this round
            if not (threshold_search_result.get_found_non_none_infeasible()
                    or coefficient_search_result.get_found_non_none_infeasible()):
                if global_best_feasible_package is not None:
                    print('Both sub-searches found no infeasible packages; returning best feasible.')
                    self.__metrics.end_execution(global_best_feasible_objective, no_of_scenarios)
                    return (global_best_feasible_package, global_best_feasible_objective, False)
                else:
                    print('No feasible package found at convergence.')
                    self.__metrics.end_execution(0, no_of_scenarios)
                    if self.__early_termination:
                        return (None, 0.0, False)

            for ind in range(len(cvar_upper_bounds)):
                if cvar_upper_bounds[ind] <\
                    cvar_lower_bounds[ind]:
                    cvar_upper_bounds[ind] -= self.__bisection_threshold
                else:
                    cvar_upper_bounds[ind] += self.__bisection_threshold

            for ind in range(len(min_no_of_scenarios_to_consider)):
                if min_no_of_scenarios_to_consider[ind] <\
                    max_no_of_scenarios_to_consider[ind]:
                    max_no_of_scenarios_to_consider[ind] -= 1

        self.__metrics.end_execution(0, no_of_scenarios)
        if self.__early_termination and global_best_feasible_package is None:
            return (None, 0.0, False)
        return (None, 0.0, True)
    

    def display_package(self, package_dict):
        if package_dict is None:
            return
        for id in package_dict:
            print(id, ',', package_dict[id])

    
    def get_metrics(self) -> OptimizationMetrics:
        return self.__metrics
