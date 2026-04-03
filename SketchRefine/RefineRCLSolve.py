import copy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import psutil

from CVaRification.CVaRificationSearchResults import CVaRificationSearchResults
from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OptimizationMetrics.OptimizationMetrics import OptimizationMetrics
from PgConnection.PgConnection import PgConnection
from SeedManager.SeedManager import SeedManager
from SketchRefine.RefineValidator import RefineValidator
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


class RefineRCLSolve:

    def __init__(
        self, query: Query,
        linear_relaxation: bool,
        chosen_partition_optimization_scenarios,
        partition_optimization_scenarios,
        chosen_tuple_optimization_scenarios,
        chosen_partition_values,
        partition_values,
        chosen_tuple_values,
        validator: RefineValidator,
        approximation_bound: float,
        bisection_threshold: float,
        partition_variable_multiplicity,
        chosen_tuple_multiplicity,
        tuple_ids,
        sketch_objective_value: float,
        check_feasibility: bool = False,
        optimize_lcvar: bool = False,
        gurobi_env = None,
        early_termination: bool = False
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
            env=self.__gurobi_env
        )
        self.__check_feasibility = check_feasibility
        self.__optimize_lcvar = optimize_lcvar
        self.__early_termination = early_termination
        if check_feasibility:
            self.__model.Params.SolutionLimit = 1
            self.__model.Params.MIPFocus = 1
        self.__is_linear_relaxation = \
            linear_relaxation
        self.__chosen_partition_optimization_scenarios =\
            chosen_partition_optimization_scenarios
        self.__partition_optimization_scenarios =\
            partition_optimization_scenarios
        self.__chosen_tuple_optimization_scenarios =\
            chosen_tuple_optimization_scenarios
        self.__chosen_partition_values =\
            chosen_partition_values
        self.__partition_values =\
            partition_values
        self.__chosen_tuple_values =\
            chosen_tuple_values
        self.__validator =\
            validator
        self.__approximation_bound =\
            approximation_bound
        self.__bisection_threshold = bisection_threshold
        self.__partition_variable_multiplicity = partition_variable_multiplicity
        self.__chosen_tuple_multiplicity = chosen_tuple_multiplicity
    
        self.__no_of_vars = 0

        if len(self.__chosen_partition_values) == 0:
            for attr in self.__chosen_partition_optimization_scenarios:
                self.__no_of_vars += len(self.__chosen_partition_optimization_scenarios[attr])
                break
        
        else:
            for attr in self.__chosen_partition_values:
                self.__no_of_vars += len(self.__chosen_partition_values[attr])
                break
            
        self.__no_of_vars += sum(
            len(self.__partition_variable_multiplicity[p])
            for p in self.__partition_variable_multiplicity
        )
        self.__no_of_vars += len(self.__chosen_tuple_multiplicity)

        self.__vars = []

        self.__risk_constraints = []
        self.__risk_to_lcvar_constraint_mapping = dict()
        self.__sorted_scenario_prefix_sums = {}
        self.__ids = tuple_ids
        self.__metrics = OptimizationMetrics('RefineRCLSolve', self.__is_linear_relaxation)
        self.__no_of_optimization_scenarios = 0

        for attr in self.__chosen_partition_optimization_scenarios:
            for tuple_scenarios in self.__chosen_partition_optimization_scenarios[attr]:
                self.__no_of_optimization_scenarios = len(tuple_scenarios)
                break
            break

        self.__sketch_objective_value = sketch_objective_value

        if self.__query.get_objective().is_cvar_objective():
            self.__V = None
            self.__Zs = []
            # Lazy Z_s constraint state
            self.__z_groups = []
            self.__z_group_constrs = []
            self.__z_lazy_active = False
    
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

    def __build_aggregate_z_lin_expr_refine(self, group: frozenset, attr: str):
        """Build aggregate Z_s constraint from three scenario sources."""
        group_list = sorted(group)
        G = len(group_list)
        lhs = gp.LinExpr([1.0] * G, [self.__Zs[s] for s in group_list])
        agg_coeffs = []
        n_chosen = len(self.__chosen_partition_optimization_scenarios[attr])
        for i in range(n_chosen):
            total = sum(
                self.__chosen_partition_optimization_scenarios[attr][i][s]
                for s in group_list
            )
            agg_coeffs.append(total)
        for partition in self.__partition_optimization_scenarios[attr]:
            n_pvars = len(
                self.__partition_optimization_scenarios[attr][partition]
            )
            for i in range(n_pvars):
                total = sum(
                    self.__partition_optimization_scenarios[attr][partition][i][s]
                    for s in group_list
                )
                agg_coeffs.append(total)
        for tup in self.__chosen_tuple_optimization_scenarios[attr]:
            n_tvars = len(
                self.__chosen_tuple_optimization_scenarios[attr][tup]
            )
            for i in range(n_tvars):
                total = sum(
                    self.__chosen_tuple_optimization_scenarios[attr][tup][i][s]
                    for s in group_list
                )
                agg_coeffs.append(total)
        rhs = gp.LinExpr(agg_coeffs, self.__vars)
        rhs.addTerms(-float(G), self.__V)
        return lhs, rhs

    def __init_z_lazy_constraints_refine(self, attr: str):
        """Initialize lazy Z_s constraints."""
        no_of_scenarios = self.__no_of_optimization_scenarios
        self.__z_groups = []
        self.__z_group_constrs = []
        all_s = frozenset(range(no_of_scenarios))
        self.__z_groups.append(all_s)
        lhs, rhs = self.__build_aggregate_z_lin_expr_refine(all_s, attr)
        tail = self.__query.get_objective().get_tail_type()
        constr = self.__model.addConstr(lhs <= rhs if tail == TailType.LOWEST else lhs >= rhs)
        self.__z_group_constrs.append(constr)
        self.__z_lazy_active = True

    def __check_z_violations_refine(self, attr: str, tol: float = 1e-6) -> list[int]:
        """Vectorized violation check for RefineRCLSolve."""
        V_val = self.__V.x
        x_arr = np.array([v.x for v in self.__vars])
        S = self.__no_of_optimization_scenarios
        rows = []
        n_chosen = len(self.__chosen_partition_optimization_scenarios[attr])
        for i in range(n_chosen):
            rows.append(
                self.__chosen_partition_optimization_scenarios[attr][i][:S]
            )
        for partition in self.__partition_optimization_scenarios[attr]:
            for i in range(len(
                    self.__partition_optimization_scenarios[attr][partition])):
                rows.append(
                    self.__partition_optimization_scenarios[attr][partition][i][:S]
                )
        for tup in self.__chosen_tuple_optimization_scenarios[attr]:
            for i in range(len(
                    self.__chosen_tuple_optimization_scenarios[attr][tup])):
                rows.append(
                    self.__chosen_tuple_optimization_scenarios[attr][tup][i][:S]
                )
        scenario_matrix = np.array(rows)
        lin_vals = scenario_matrix.T @ x_arr
        rhs_vals = lin_vals - V_val
        z_vals = np.array([z.x for z in self.__Zs])
        tail = self.__query.get_objective().get_tail_type()
        if tail == TailType.LOWEST:
            mask = z_vals > rhs_vals + tol
        else:
            mask = z_vals < rhs_vals - tol
        return list(np.where(mask)[0])

    def __refine_z_constraints_refine(self, attr: str, tol: float = 1e-6) -> bool:
        """Perform one round of constraint splitting."""
        violated_set = set(self.__check_z_violations_refine(attr, tol))
        if not violated_set:
            return False
        tail = self.__query.get_objective().get_tail_type()
        new_groups, new_constrs = [], []
        for group, constr in zip(self.__z_groups, self.__z_group_constrs):
            viol = group & violated_set
            sat  = group - violated_set
            if not viol:
                new_groups.append(group)
                new_constrs.append(constr)
                continue
            self.__model.remove(constr)
            for sub in [sat, viol]:
                if not sub:
                    continue
                lhs, rhs = self.__build_aggregate_z_lin_expr_refine(sub, attr)
                c = self.__model.addConstr(lhs <= rhs if tail == TailType.LOWEST else lhs >= rhs)
                new_groups.append(sub)
                new_constrs.append(c)
        self.__z_groups = new_groups
        self.__z_group_constrs = new_constrs
        return True

    def __get_upper_bound_for_repetition(self) -> int:
        for constraint in self.__query.get_constraints():
            if constraint.is_repeat_constraint():
                return 1+constraint.get_repetition_limit()
        return None
    
    def __add_variables_to_model(self) -> None:
        max_repetition = self.__get_upper_bound_for_repetition()
        type = GRB.INTEGER
        if self.__is_linear_relaxation:
            type = GRB.CONTINUOUS
        self.__vars = []
        if len(self.__chosen_partition_optimization_scenarios) == 0:
            for attr in self.__chosen_partition_values:
                for _ in range(len(self.__chosen_partition_values[attr])):
                    if max_repetition is not None:
                        self.__vars.append(self.__model.addVar(
                            lb=0, ub=max_repetition, vtype=type))
                    else:
                        self.__vars.append(self.__model.addVar(
                            lb=0, vtype=type))
                break
        else:
            for attr in self.__chosen_partition_optimization_scenarios:
                for _ in range(len(
                    self.__chosen_partition_optimization_scenarios[attr])):
                    if max_repetition is not None:
                        self.__vars.append(self.__model.addVar(
                            lb=0, ub=max_repetition, vtype=type))
                    else:
                        self.__vars.append(self.__model.addVar(
                            lb=0, vtype=type))
                break
        
        for partition in self.__partition_variable_multiplicity:
            for multiplicity in self.__partition_variable_multiplicity[partition]:
                self.__vars.append(self.__model.addVar(
                    lb=multiplicity, ub=multiplicity, vtype=type))

        for tuple_id in self.__chosen_tuple_multiplicity:
            multiplicity = self.__chosen_tuple_multiplicity[tuple_id]
            self.__vars.append(self.__model.addVar(
                lb=multiplicity, ub=multiplicity, vtype=type))

        if not self.__check_feasibility and not self.__optimize_lcvar and \
                self.__query.get_objective().is_cvar_objective():
            self.__V = self.__model.addVar(vtype=GRB.CONTINUOUS)
            self.__Zs = []
            for _ in range(self.__no_of_optimization_scenarios):
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
        size_limit = package_size_constraint.get_package_size_limit()
        gurobi_inequality = self.__get_gurobi_inequality(
            package_size_constraint.get_inequality_sign())

        self.__model.addConstr(
            gp.LinExpr([1]*self.__no_of_vars, self.__vars),
            gurobi_inequality, size_limit
        )
    
    def __add_deterministic_constraint_to_model(
        self, deterministic_constraint: DeterministicConstraint
    ):
        attribute = deterministic_constraint.get_attribute_name()
        gurobi_inequality = self.__get_gurobi_inequality(
            deterministic_constraint.get_inequality_sign())
        sum_limit = deterministic_constraint.get_sum_limit()

        values = list(self.__chosen_partition_values[attribute])

        for partition in self.__partition_values[attribute]:
            for value in self.__partition_values[attribute][partition]:
                values.append(value)
        
        for tuple in self.__chosen_tuple_values[attribute]:
            for value in self.__chosen_tuple_values[attribute][tuple]:
                values.append(value[0])

        self.__model.addConstr(
            gp.LinExpr(values, self.__vars), gurobi_inequality,
            sum_limit
        )
    
    def __add_expected_sum_constraint_to_model(
        self, expected_sum_constraint: ExpectedSumConstraint,
        threshold_override: float | None = None
    ):
        attr = expected_sum_constraint.get_attribute_name()
        coefficients = []

        for scenarios in self.__chosen_partition_optimization_scenarios[attr]:
            coefficients.append(np.average(scenarios))
        
        for partition in self.__partition_optimization_scenarios[attr]:
            for scenarios in self.__partition_optimization_scenarios[attr][partition]:
                coefficients.append(np.average(scenarios))
        
        for tuple in self.__chosen_tuple_optimization_scenarios[attr]:
            for scenarios in self.__chosen_tuple_optimization_scenarios[attr][tuple]:
                coefficients.append(np.average(scenarios))
        
        gurobi_inequality = self.__get_gurobi_inequality(
            expected_sum_constraint.get_inequality_sign()
        )

        expected_sum_limit = threshold_override
        if expected_sum_limit is None:
            expected_sum_limit = expected_sum_constraint.get_sum_limit()

        return self.__model.addConstr(
            gp.LinExpr(coefficients, self.__vars),
            gurobi_inequality, expected_sum_limit)

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
            n = self.__no_of_optimization_scenarios
            k_arr = np.arange(1, n + 1, dtype=np.float64)
            for var_idx in range(
                    len(self.__chosen_partition_optimization_scenarios[attr])):
                arr = np.array(
                    self.__chosen_partition_optimization_scenarios[attr][var_idx],
                    dtype=np.float64)
                sorted_arr = np.sort(arr) if tail_type == TailType.LOWEST \
                    else np.sort(arr)[::-1]
                prefix_avgs.append(np.cumsum(sorted_arr) / k_arr)
            for partition in self.__partition_optimization_scenarios[attr]:
                for var_idx in range(
                        len(self.__partition_optimization_scenarios[attr][partition])):
                    arr = np.array(
                        self.__partition_optimization_scenarios[attr][partition][var_idx],
                        dtype=np.float64)
                    sorted_arr = np.sort(arr) if tail_type == TailType.LOWEST \
                        else np.sort(arr)[::-1]
                    prefix_avgs.append(np.cumsum(sorted_arr) / k_arr)
            for tup in self.__chosen_tuple_optimization_scenarios[attr]:
                for var_idx in range(
                        len(self.__chosen_tuple_optimization_scenarios[attr][tup])):
                    arr = np.array(
                        self.__chosen_tuple_optimization_scenarios[attr][tup][var_idx],
                        dtype=np.float64)
                    sorted_arr = np.sort(arr) if tail_type == TailType.LOWEST \
                        else np.sort(arr)[::-1]
                    prefix_avgs.append(np.cumsum(sorted_arr) / k_arr)
            self.__sorted_scenario_prefix_sums[key] = prefix_avgs
        if self.__optimize_lcvar and self.__query.get_objective().is_cvar_objective():
            obj = self.__query.get_objective()
            key = (obj.get_attribute_name(), obj.get_tail_type())
            if key not in self.__sorted_scenario_prefix_sums:
                attr = obj.get_attribute_name()
                tail_type = obj.get_tail_type()
                n = self.__no_of_optimization_scenarios
                k_arr = np.arange(1, n + 1, dtype=np.float64)
                prefix_avgs = []
                for var_idx in range(
                        len(self.__chosen_partition_optimization_scenarios[attr])):
                    arr = np.array(
                        self.__chosen_partition_optimization_scenarios[attr][var_idx],
                        dtype=np.float64)
                    sorted_arr = np.sort(arr) if tail_type == TailType.LOWEST \
                        else np.sort(arr)[::-1]
                    prefix_avgs.append(np.cumsum(sorted_arr) / k_arr)
                for partition in self.__partition_optimization_scenarios[attr]:
                    for var_idx in range(
                            len(self.__partition_optimization_scenarios[attr][partition])):
                        arr = np.array(
                            self.__partition_optimization_scenarios[attr][partition][var_idx],
                            dtype=np.float64)
                        sorted_arr = np.sort(arr) if tail_type == TailType.LOWEST \
                            else np.sort(arr)[::-1]
                        prefix_avgs.append(np.cumsum(sorted_arr) / k_arr)
                for tup in self.__chosen_tuple_optimization_scenarios[attr]:
                    for var_idx in range(
                            len(self.__chosen_tuple_optimization_scenarios[attr][tup])):
                        arr = np.array(
                            self.__chosen_tuple_optimization_scenarios[attr][tup][var_idx],
                            dtype=np.float64)
                        sorted_arr = np.sort(arr) if tail_type == TailType.LOWEST \
                            else np.sort(arr)[::-1]
                        prefix_avgs.append(np.cumsum(sorted_arr) / k_arr)
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
            tuple_wise_heaps.append(Heap(is_max_heap))

        n_chosen_partition = len(self.__chosen_partition_optimization_scenarios[attr])
        partition_offsets = {}
        var_offset = n_chosen_partition
        for partition in self.__partition_optimization_scenarios[attr]:
            partition_offsets[partition] = var_offset
            var_offset += len(self.__partition_optimization_scenarios[attr][partition])
        tuple_offsets = {}
        for tuple in self.__chosen_tuple_optimization_scenarios[attr]:
            tuple_offsets[tuple] = var_offset
            var_offset += len(self.__chosen_tuple_optimization_scenarios[attr][tuple])

        for scenario_index in range(total_no_of_scenarios):
            for var in range(n_chosen_partition):
                scenario_value = self.__chosen_partition_optimization_scenarios[
                    attr][var][scenario_index]
                heap_idx = var
                tuple_wise_heaps[heap_idx].push(scenario_value)
                if tuple_wise_heaps[heap_idx].size() > no_of_scenarios_to_consider:
                    tuple_wise_heaps[heap_idx].pop()

            for partition in self.__partition_optimization_scenarios[attr]:
                for var in range(len(self.__partition_optimization_scenarios[attr][partition])):
                    scenario_value = self.__partition_optimization_scenarios[
                        attr][partition][var][scenario_index]
                    heap_idx = partition_offsets[partition] + var
                    tuple_wise_heaps[heap_idx].push(scenario_value)
                    if tuple_wise_heaps[heap_idx].size() > no_of_scenarios_to_consider:
                        tuple_wise_heaps[heap_idx].pop()

            for tuple in self.__chosen_tuple_optimization_scenarios[attr]:
                for var in range(len(self.__chosen_tuple_optimization_scenarios[attr][tuple])):
                    scenario_value = self.__chosen_tuple_optimization_scenarios[
                        attr][tuple][var][scenario_index]
                    heap_idx = tuple_offsets[tuple] + var
                    tuple_wise_heaps[heap_idx].push(scenario_value)
                    if tuple_wise_heaps[heap_idx].size() > no_of_scenarios_to_consider:
                        tuple_wise_heaps[heap_idx].pop()

        return [tuple_wise_heaps[var].sum()/no_of_scenarios_to_consider\
                for var in range(self.__no_of_vars)]

    def __add_lcvar_constraint_to_model(
        self, risk_constraint: CVaRConstraint | VaRConstraint,
        cvarified_constraint: CVaRConstraint,
        no_of_scenarios: int, no_of_scenarios_to_consider: int
    ):
        coefficients = \
            self.__get_cvar_constraint_coefficients(
                cvarified_constraint, no_of_scenarios_to_consider,
                no_of_scenarios
            )
        
        gurobi_inequality = self.__get_gurobi_inequality(
            cvarified_constraint.get_inequality_sign())


        self.__risk_constraints.append(risk_constraint)
        self.__risk_to_lcvar_constraint_mapping[risk_constraint] =\
            self.__model.addConstr(
                gp.LinExpr(coefficients, self.__vars), gurobi_inequality,
                cvarified_constraint.get_sum_limit()
            )
    
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
                self.__add_package_size_constraint_to_model(constraint)
            if constraint.is_deterministic_constraint():
                self.__add_deterministic_constraint_to_model(constraint)
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
                    if stochastic_constraint_index not in trivial_constraints:
                        if constraint.is_cvar_constraint():
                            cvarified_constraint = constraint
                        else:
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint=constraint,
                                    cvar_threshold=cvar_thresholds[
                                        stochastic_constraint_index]
                                )
                        
                        self.__add_lcvar_constraint_to_model(
                            risk_constraint=constraint,
                            cvarified_constraint=cvarified_constraint,
                            no_of_scenarios=no_of_scenarios,
                            no_of_scenarios_to_consider=\
                                no_of_scenarios_to_consider[
                                    stochastic_constraint_index]
                        )

                else:
                    if not any_high_tail_cvar:
                        attribute = constraint.get_attribute_name()
                        sum_limit = constraint.get_sum_limit()
                        coefficients = []

                        for tuple_scenarios in self.__chosen_partition_optimization_scenarios[attribute]:
                            coefficients.append(np.median(tuple_scenarios))

                        for partition in self.__partition_optimization_scenarios[attribute]:
                            for tuple_scenarios in self.__partition_optimization_scenarios[attribute][partition]:
                                coefficients.append(np.median(tuple_scenarios))

                        for tuple in self.__chosen_tuple_optimization_scenarios[attribute]:
                            for tuple_scenarios in self.__chosen_tuple_optimization_scenarios[attribute][tuple]:
                                coefficients.append(np.median(tuple_scenarios))

                        gurobi_inequality = GRB.LESS_EQUAL
                        if constraint.get_inequality_sign() == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                            gurobi_inequality = GRB.GREATER_EQUAL

                        self.__model.addConstr(
                            gp.LinExpr(coefficients, self.__vars), gurobi_inequality, sum_limit)

                stochastic_constraint_index += 1

        if not self.__check_feasibility and not self.__optimize_lcvar and \
                self.__query.get_objective().is_cvar_objective():
            attr = self.__query.get_objective().get_attribute_name()
            self.__z_lazy_active = False
            self.__init_z_lazy_constraints_refine(attr)

    def __add_objective_to_model(
        self, objective: Objective,
        no_of_scenarios: int
    ):
        if self.__check_feasibility:
            if objective.is_cvar_objective():
                attr = objective.get_attribute_name()
                coefficients = []
                for idx in range(len(self.__chosen_partition_optimization_scenarios[attr])):
                    coefficients.append(
                        np.average(self.__chosen_partition_optimization_scenarios[attr][idx]))
                for partition in self.__partition_optimization_scenarios[attr]:
                    for scenario in self.__partition_optimization_scenarios[attr][partition]:
                        coefficients.append(np.average(scenario))
                for tuple_id in self.__chosen_tuple_optimization_scenarios[attr]:
                    for scenario in self.__chosen_tuple_optimization_scenarios[attr][tuple_id]:
                        coefficients.append(np.average(scenario))
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

        if objective.get_stochasticity() ==\
            Stochasticity.DETERMINISTIC:
            coefficients = list(self.__chosen_partition_values[attr])
            for partition in self.__partition_values[attr]:
                for value in self.__partition_values[attr][partition]:
                    coefficients.append(value)

            for tuple_id in self.__chosen_tuple_values[attr]:
                for value in self.__chosen_tuple_values[attr][tuple_id]:
                    coefficients.append(value[0])
        else:
            for idx in range(len(self.__chosen_partition_optimization_scenarios[attr])):
                coefficients.append(
                    np.average(self.__chosen_partition_optimization_scenarios[attr][idx]))
            
            for partition in self.__partition_optimization_scenarios[attr]:
                for scenario in self.__partition_optimization_scenarios[attr][partition]:
                    coefficients.append(np.average(scenario))
            
            for tuple in self.__chosen_tuple_optimization_scenarios[attr]:
                for scenario in self.__chosen_tuple_optimization_scenarios[attr][tuple]:
                    coefficients.append(np.average(scenario))
        
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
        self.__model = gp.Model(env=self.__gurobi_env)
        if self.__check_feasibility:
            self.__model.Params.SolutionLimit = 1
            self.__model.Params.MIPFocus = 1
        self.__add_variables_to_model()
        self.__add_constraints_to_model(
            no_of_scenarios,
            no_of_scenarios_to_consider,
            probabilistically_constrained,
            cvar_lower_bounds, trivial_constraints)
        self.__add_objective_to_model(
            self.__query.get_objective(), no_of_scenarios)
    
    def __extract_package_from_model(self):
        """Extract solution from model variables."""
        package_dict = {}
        try:
            for idx in range(len(self.__ids)):
                var = self.__vars[idx]
                if var.x > 0:
                    package_dict[self.__ids[idx]] = var.x
        except AttributeError:
            return None
        if not package_dict:
            return None
        return package_dict

    def __get_package_with_z_refinement_refine(self):
        """Solve with lazy Z_s constraint refinement."""
        attr = self.__query.get_objective().get_attribute_name()
        max_rounds = self.__no_of_optimization_scenarios + 1
        for _ in range(max_rounds):
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
            if not self.__refine_z_constraints_refine(attr):
                break
        return self.__extract_package_from_model()

    def __get_package(self):
        if (not self.__check_feasibility and not self.__optimize_lcvar
                and self.__query.get_objective().is_cvar_objective()
                and self.__z_lazy_active):
            return self.__get_package_with_z_refinement_refine()
        self.__metrics.start_optimizer()
        self.__model.optimize()
        self.__metrics.end_optimizer()
        return self.__extract_package_from_model()

    def __get_package_with_indices(self, package_dict):
        if package_dict is None:
            return None
        
        package_dict_with_indices = dict()
        
        for idx in range(len(self.__ids)):
            if self.__ids[idx] in package_dict:
                package_dict_with_indices[idx] = \
                    package_dict[self.__ids[idx]]
        
        return package_dict_with_indices

    def __is_var_constraint_satisfied(
        self, var_constraint: VaRConstraint,
        var_validation: float
    ) -> bool:
        if var_constraint.get_inequality_sign() ==\
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            return var_validation >=\
                var_constraint.get_sum_limit()
        
        return var_validation <= \
            var_constraint.get_sum_limit()

    def __is_cvar_constraint_satisfied(
        self, cvar_constraint: CVaRConstraint,
        cvar_validation: float
    ) -> bool:
        if cvar_constraint.get_inequality_sign() ==\
            RelationalOperators.LESS_THAN_OR_EQUAL_TO:
            return cvar_validation <= \
                cvar_constraint.get_sum_limit()
        return cvar_validation >= \
            cvar_constraint.get_sum_limit()

    def __l_inf(
        self, l1: list[int | float],
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

            package_with_indices = package
            unacceptable_diff, _ = \
                self.__is_objective_value_relative_diff_high(
                    package, package_with_indices
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
        no_of_scenarios_to_consider: list[int],
        is_model_setup: bool
    ):
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
            cvar_upper_bounds, cvar_lower_bounds) >=\
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
                    if stochastic_constraint_index not in\
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
            
            all_constraints_satisfied = True
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
                            constraint, cvar_validation):
                            print('CVaR constraint satisfied')
                            cvar_lower_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                        else:
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
                            print('VaR constraint is satisfied')
                            cvar_lower_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                        else:
                            print('VaR constraint not satisfied')
                            all_constraints_satisfied = False
                            cvar_upper_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]

                stochastic_constraint_index += 1

            if all_constraints_satisfied:
                if self.__check_feasibility or \
                        self.__validator.is_package_1_pm_epsilon_approximate(
                    package, epsilon=self.__approximation_bound,
                    upper_bound=self.__sketch_objective_value
                ):
                    result = CVaRificationSearchResults()
                    result.set_found_appropriate_package(True)
                    result.set_package(package)
                    result.set_objective_value(
                        self.__validator.get_validation_objective_value(package))
                    return result
                else:
                    # Package satisfies constraints but objective not good enough
                    validation_obj = self.__validator.get_validation_objective_value(package)
                    if best_feasible_package is None or \
                        (self.__query.get_objective().get_objective_type() == \
                            ObjectiveType.MAXIMIZATION and validation_obj > best_feasible_objective) or \
                        (self.__query.get_objective().get_objective_type() != \
                            ObjectiveType.MAXIMIZATION and validation_obj < best_feasible_objective):
                        best_feasible_package = package
                        best_feasible_objective = validation_obj
            else:
                # Package is not None but constraints are not satisfied
                if package is not None:
                    found_non_none_infeasible = True

        result = CVaRificationSearchResults()
        result.set_objective_upper_bound(self.__sketch_objective_value)
        result.set_cvar_thresholds(cvar_lower_bounds)
        if best_feasible_package is not None:
            result.set_best_feasible_package(best_feasible_package)
            result.set_best_feasible_objective(best_feasible_objective)
        result.set_found_non_none_infeasible(found_non_none_infeasible)
        return result
    
    def __cvar_coefficient_search(
        self, no_of_scenarios: int,
        cvar_thresholds: list[float | int],
        trivial_constraints: list[int],
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
            max_no_of_scenarios_to_consider) > 1:

            mid_no_of_scenarios_to_consider = []
            for ind in range(len(min_no_of_scenarios_to_consider)):
                mid_no_of_scenarios_to_consider.append(
                    (min_no_of_scenarios_to_consider[ind] +\
                        max_no_of_scenarios_to_consider[ind]) // 2
                )
            
            if not is_model_setup:
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
                stochastic_constraint_index = 0
                for constraint in self.__get_searchable_constraints():
                    if constraint.is_risk_constraint() and \
                            stochastic_constraint_index not in trivial_constraints:
                        if constraint.is_cvar_constraint():
                            cvarified_constraint = constraint
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
                    stochastic_constraint_index += 1
            
            package = self.__get_package()
            if package is None:
                min_no_of_scenarios_to_consider = \
                    mid_no_of_scenarios_to_consider
                continue
                
            all_constraints_satisfied = True
            stochastic_constraint_index = 0

            for constraint in self.__get_searchable_constraints():
                is_trivial = stochastic_constraint_index in trivial_constraints
                if constraint.is_expected_sum_constraint():
                    expected_sum_satisfied = \
                        self.__validator.get_expected_sum_constraint_feasibility(
                            package, constraint
                        )
                    if not expected_sum_satisfied:
                        all_constraints_satisfied = False
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
                            min_no_of_scenarios_to_consider[stochastic_constraint_index] =\
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index]
                        else:
                            all_constraints_satisfied = False
                            max_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index] - 1
                    
                    if constraint.is_var_constraint():
                        var_validation = self.__validator.get_var_among_validation_scenarios(
                            package, constraint
                        )
                        print('VaR validation:', var_validation)

                        if self.__is_var_constraint_satisfied(constraint, var_validation):
                            min_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index]
                        
                        else:
                            all_constraints_satisfied = False
                            max_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index] - 1
                stochastic_constraint_index += 1
            
            if all_constraints_satisfied:
                if self.__check_feasibility or \
                        self.__validator.is_package_1_pm_epsilon_approximate(
                    package, epsilon=self.__approximation_bound,
                    upper_bound=self.__sketch_objective_value
                ):
                    result = CVaRificationSearchResults()
                    result.set_found_appropriate_package(True)
                    result.set_package(package)
                    result.set_objective_value(
                        self.__validator.get_validation_objective_value(package))
                    return result
                else:
                    # Package satisfies constraints but objective not good enough
                    validation_obj = self.__validator.get_validation_objective_value(package)
                    if best_feasible_package is None or \
                        (self.__query.get_objective().get_objective_type() == \
                            ObjectiveType.MAXIMIZATION and validation_obj > best_feasible_objective) or \
                        (self.__query.get_objective().get_objective_type() != \
                            ObjectiveType.MAXIMIZATION and validation_obj < best_feasible_objective):
                        best_feasible_package = package
                        best_feasible_objective = validation_obj
            else:
                # Package is not None but constraints are not satisfied
                if package is not None:
                    found_non_none_infeasible = True

        result = CVaRificationSearchResults()
        result.set_objective_upper_bound(self.__sketch_objective_value)
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
        no_of_scenarios_to_consider =\
            self.__get_no_of_scenarios_to_consider(
                cvar_constraint, no_of_scenarios
            )

        coefficients = self.__get_cvar_constraint_coefficients(
            cvar_constraint, no_of_scenarios_to_consider,
            no_of_scenarios
        )

        linearized_cvar = 0
        idx = 0
        for id in self.__ids:
            if id in probabilistically_unconstrained_package:
                linearized_cvar += coefficients[idx] * \
                    probabilistically_unconstrained_package[id]
            idx += 1
        
        for partition in self.__partition_variable_multiplicity:
            for multiplicity in self.__partition_variable_multiplicity[partition]:
                linearized_cvar += coefficients[idx] * multiplicity
                idx += 1

        for tuple_id in self.__chosen_tuple_multiplicity:
            linearized_cvar += coefficients[idx] * \
                self.__chosen_tuple_multiplicity[tuple_id]
            idx += 1

        return linearized_cvar

    def __get_expected_sum_among_optimization_scenarios(
        self, probabilistically_unconstrained_package,
        attribute
    ):
        expected_sum = 0
        for idx, id in enumerate(self.__ids):
            if id in probabilistically_unconstrained_package:
                expected_sum += np.mean(
                    self.__chosen_partition_optimization_scenarios[attribute][idx])*\
                        probabilistically_unconstrained_package[id]
        for partition in self.__partition_optimization_scenarios[attribute]:
            for dup_idx, scenarios in enumerate(
                self.__partition_optimization_scenarios[attribute][partition]
            ):
                expected_sum += np.mean(scenarios)*\
                    self.__partition_variable_multiplicity[partition][dup_idx]
        for tuple_id in self.__chosen_tuple_optimization_scenarios[attribute]:
            for scenarios in self.__chosen_tuple_optimization_scenarios[attribute][tuple_id]:
                expected_sum += np.mean(scenarios)*\
                    self.__chosen_tuple_multiplicity[tuple_id]
        return expected_sum


    def __get_bounds_for_risk_constraints(
        self, no_of_scenarios, probabilistically_unconstrained_package
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
                    probabilistically_unconstrained_package, constraint
                ):
                    is_satisfied = True
            elif constraint.is_var_constraint():
                if self.__validator.get_var_constraint_feasibility(
                    probabilistically_unconstrained_package, constraint
                ):
                    is_satisfied = True
            elif constraint.is_cvar_constraint():
                if self.__validator.get_cvar_constraint_feasibility(
                    probabilistically_unconstrained_package, constraint
                ):
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
                cvar_lower_bounds.append(expected_sum)
                
                no_of_scenarios_to_consider = \
                    max(1, int(np.floor(constraint.get_percentage_of_scenarios()\
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
                        constraint.get_attribute_name()
                    )
                )

                no_of_scenarios_to_consider =\
                    max(1, int(np.floor(
                        cvarified_constraint.get_percentage_of_scenarios()\
                         *no_of_scenarios/100.0)))

                min_no_of_scenarios_to_consider.append(no_of_scenarios_to_consider)
                max_no_of_scenarios_to_consider.append(no_of_scenarios)
            
            stochastic_constraint_index += 1
        
        for idx in range(len(cvar_lower_bounds)):
            if cvar_lower_bounds[idx] > cvar_upper_bounds[idx]:
                cvar_upper_bounds[idx] -= cvar_upper_bounds[idx]
            else:
                cvar_upper_bounds[idx] += cvar_upper_bounds[idx]
        return cvar_upper_bounds, cvar_lower_bounds, max_no_of_scenarios_to_consider,\
            min_no_of_scenarios_to_consider, trivial_constraints


    def solve(self, can_add_scenarios = False):
        self.__metrics.start_execution()
        no_of_scenarios = self.__no_of_optimization_scenarios

        self.__model_setup(
            no_of_scenarios=no_of_scenarios,
            no_of_scenarios_to_consider=[],
            probabilistically_constrained=False
        )

        probabilistically_unconstrained_package = \
            self.__get_package()
        print('Probabilistically unconstrainted package:',
            probabilistically_unconstrained_package)
        if probabilistically_unconstrained_package is None:
            self.__metrics.end_execution(0, 0)
            return (None, 0.0, False)
        
        if self.__validator.is_package_validation_feasible(
            probabilistically_unconstrained_package):
            print('Probabilistically unconstrained package'
                'is validation feasible.')
            self.__metrics.end_execution(
                self.__validator.get_validation_objective_value(
                    probabilistically_unconstrained_package),
                no_of_scenarios
            )
            return (
                probabilistically_unconstrained_package,
                self.__validator.get_validation_objective_value(
                    probabilistically_unconstrained_package
                ), False
            )
        
        cvar_upper_bounds, cvar_lower_bounds,\
        max_no_of_scenarios_to_consider,\
        min_no_of_scenarios_to_consider,\
        trivial_constraints = \
            self.__get_bounds_for_risk_constraints(
                no_of_scenarios,
                probabilistically_unconstrained_package
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

        while self.__l_inf(cvar_upper_bounds, cvar_lower_bounds) >=\
            self.__bisection_threshold*init_diff or\
            self.__l_inf(min_no_of_scenarios_to_consider,
                         max_no_of_scenarios_to_consider) >= 1:
            
            print('CVaR upper bounds:', cvar_upper_bounds)
            print('CVaR lower bounds:', cvar_lower_bounds)
            print('Min No of Scenarios to consider:',
                min_no_of_scenarios_to_consider)
            print('Max no of scenarios to consider:',
                max_no_of_scenarios_to_consider)

            threshold_search_result = \
                self.__cvar_threshold_search(
                    no_of_scenarios,
                    cvar_upper_bounds,
                    cvar_lower_bounds,
                    trivial_constraints,
                    min_no_of_scenarios_to_consider,
                    is_model_setup
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

            coefficient_search_result = \
                self.__cvar_coefficient_search(
                    no_of_scenarios,
                    cvar_upper_bounds,
                    trivial_constraints,
                    min_no_of_scenarios_to_consider,
                    max_no_of_scenarios_to_consider,
                    is_model_setup,
                    can_add_scenarios,
                )

            if coefficient_search_result.\
                get_needs_more_scenarios():
                return (None, 0.0, True)

            if coefficient_search_result.\
                get_found_appropriate_package():
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
