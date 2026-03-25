import gurobipy as gp
from gurobipy import GRB
import numpy as np

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
        partition_id_for_each_index: list[int]
    ):
        self.__query = query
        self.__gurobi_env = gp.Env(
            params=GurobiLicense.OPTIONS)
        self.__gurobi_env.setParam(
            'OutputFlag', 0
        )
        self.__model = gp.Model(
            env=self.__gurobi_env)
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
        self.__validator.update_partition_id_in_duplicate_vector(
            partition_id_in_duplicate_vector
        )
    
    def update_scenarios(
        self, updated_scenarios, no_of_scenarios
    ):
        self.__scenarios = updated_scenarios
        self.__no_of_scenarios = no_of_scenarios
        self.__opt_scenario_scores_cache = {}

    def __get_upper_bound_for_repetitions(self) -> int:
        for constraint in self.__query.get_constraints():
            if constraint.is_repeat_constraint():
                return 1 + constraint.get_repetition_limit()
        return None
    
    
    def __add_variables_to_model(self) -> None:
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

        self.__model.addConstr(
            gp.LinExpr([1]*self.__no_of_vars, self.__vars),
            gurobi_inequality, size_limit
        )
    

    def __add_deterministic_constraint_to_model(
        self, deterministic_constraint: DeterministicConstraint
    ):
        attribute = deterministic_constraint.get_attribute_name()
        gurobi_inequality = \
            self.__get_gurobi_inequality(
                deterministic_constraint.get_inequality_sign())
        sum_limit = deterministic_constraint.get_sum_limit()
        
        self.__model.addConstr(
            gp.LinExpr(self.__values[attribute], self.__vars),
            gurobi_inequality, sum_limit
        )


    def __add_expected_sum_constraint_to_model(
        self, expected_sum_constraint: ExpectedSumConstraint,
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

        expected_sum_limit = \
            expected_sum_constraint.get_sum_limit()

        self.__model.addConstr(
            gp.LinExpr(coefficients, self.__vars),
            gurobi_inequality, expected_sum_limit
        )
    

    def __get_no_of_scenarios_to_consider(
        self, cvar_constraint: CVaRConstraint,
        no_of_scenarios: int
    ):
        return int(np.floor(
            (cvar_constraint.get_percentage_of_scenarios()\
                *no_of_scenarios)/100))


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
                self.__model.addConstr(
                    gp.LinExpr(
                        coefficients, self.__vars),
                    gurobi_inequality,
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
        risk_constraint_index = 0
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
                print('Adding expected sum constraint to model')
                self.__add_expected_sum_constraint_to_model(
                    constraint
                )
            if constraint.is_risk_constraint():
                if probabilistically_constrained:
                    print('Considering risk constraint', risk_constraint_index)
                    if risk_constraint_index not in trivial_constraints:
                        print('Risk constraint is not trivial')
                        if constraint.is_cvar_constraint():
                            cvarified_constraint = constraint
                        else:
                            print('CVaRifying VaR constraint')
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint=constraint,
                                    cvar_threshold=cvar_thresholds[
                                        risk_constraint_index]
                                )
                        
                        print('Adding LCVaR constraint')
                        self.__add_lcvar_constraint_to_model(
                            risk_constraint=constraint,
                            cvarified_constraint=cvarified_constraint,
                            no_of_scenarios=no_of_scenarios,
                            no_of_scenarios_to_consider=\
                                no_of_scenarios_to_consider[
                                    risk_constraint_index
                                ]
                        )
                else:
                    print('Probabilistically unconstrained problem')
                    print('Adding weaker constraint')
                    attribute = constraint.get_attribute_name()
                    sum_limit = constraint.get_sum_limit()
                    coefficients = []
                    for tuple_scenarios in self.__scenarios[attribute]:
                        coefficients.append(np.median(tuple_scenarios))
                        '''
                        if constraint.get_inequality_sign() == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                            coefficients.append(np.max(tuple_scenarios))
                        else:
                            coefficients.append(np.min(tuple_scenarios))
                        '''
                    gurobi_inequality = GRB.LESS_EQUAL
                    if constraint.get_inequality_sign() == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                        gurobi_inequality = GRB.GREATER_EQUAL
                    self.__model.addConstr(
                        gp.LinExpr(coefficients, self.__vars), gurobi_inequality, sum_limit)
                
                risk_constraint_index += 1
                

    
    def __add_objective_to_model(
        self, objective: Objective,
        no_of_scenarios: int):

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
        print('Adding Variables to model')
        self.__add_variables_to_model()
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

    def __get_package(self):
        self.__metrics.start_optimizer()
        self.__model.optimize()
        self.__metrics.end_optimizer()
        package_dict = {}
        idx = 0
        try:
            for var in self.__vars:
                if var.x > 0:
                    package_dict[
                        self.__ids[idx]] = \
                            var.x
                idx += 1
        except AttributeError:
            return None
        if not package_dict:
            return None
        return package_dict


    def __get_package_with_indices(self):
        return self.__get_package()
    
    def __get_package_tuple_partitions(self):
        package_dict = self.__get_package()
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
        attr = self.__query.get_objective().get_attribute_name()
        
        sum = 0
        for idx in package_with_indices:
            sum += np.average(self.__scenarios[attr][idx]) * \
                package_with_indices[idx]
        
        return sum
    

    def __is_objective_value_relative_diff_high(
        self, package, package_with_indices,
    ):
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

        rel_diff = diff / (objective_value_validation_scenarios
            + 0.00001)
        
        if rel_diff > self.__sampling_tolerance:
            return True, objective_value_validation_scenarios
        
        return False, objective_value_validation_scenarios
    

    def __is_objective_value_enough(
        self, objective_value: float,
        objective_upper_bound: float
    ):
        objective = self.__query.get_objective()

        if objective.get_objective_type() == \
            ObjectiveType.MAXIMIZATION:
            return objective_value >= \
                (1 - self.__approximation_bound) *\
                    objective_upper_bound
        
        return objective_value <= \
            (1 + self.__approximation_bound) *\
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
                risk_constraint_index = 0
                for constraint in self.__query.get_constraints():
                    if constraint.is_risk_constraint():
                        if risk_constraint_index not in \
                            trivial_constraints:
                            self.__risk_to_lcvar_constraint_mapping[
                                constraint].rhs = cvar_mid_thresholds[
                                    risk_constraint_index]
                        risk_constraint_index += 1
            
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
            risk_constraint_index = 0
            
            for constraint in self.__query.get_constraints():
                if constraint.is_risk_constraint():
                    if risk_constraint_index not in trivial_constraints:
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
                                cvar_lower_bounds[risk_constraint_index] = \
                                    cvar_mid_thresholds[risk_constraint_index]
                            else:
                                print('CVaR constraint not satisfied')
                                all_constraints_satisfied = False
                                cvar_upper_bounds[risk_constraint_index] = \
                                    cvar_mid_thresholds[risk_constraint_index]
                    
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
                                cvar_lower_bounds[risk_constraint_index] = \
                                    cvar_mid_thresholds[risk_constraint_index]
                            else:
                                print('VaR constraint not satisfied')
                                all_constraints_satisfied = False
                                cvar_upper_bounds[risk_constraint_index] = \
                                    cvar_mid_thresholds[risk_constraint_index]
                    risk_constraint_index += 1

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
                risk_constraint_index = 0
                for constraint in self.__query.get_constraints():
                    if constraint.is_risk_constraint():
                        if risk_constraint_index not in trivial_constraints:
                            print('Risk constraint', risk_constraint_index,
                                  'is not trivial')
                            print('Getting constraints with',
                                  mid_no_of_scenarios_to_consider[risk_constraint_index],
                                  'scenarios from each tuple')
                            if constraint.is_cvar_constraint():
                                cvarified_constraint = constraint
                            else:
                                cvarified_constraint = \
                                    self.__get_cvarified_constraint(
                                        constraint, cvar_thresholds[
                                            risk_constraint_index])
                            coefficients = \
                                self.__get_cvar_constraint_coefficients(
                                    cvarified_constraint,
                                    mid_no_of_scenarios_to_consider[
                                        risk_constraint_index],
                                    no_of_scenarios)
                        
                            for _ in range(len(self.__vars)):
                                self.__model.chgCoeff(
                                    self.__risk_to_lcvar_constraint_mapping[
                                        constraint],
                                    self.__vars[_], coefficients[_]
                                )
                            print('Changing constraint coefficients')                    
                        risk_constraint_index += 1
            
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
            risk_constraint_index = 0

            for constraint in self.__query.get_constraints():
                if constraint.is_risk_constraint():
                    if risk_constraint_index not in trivial_constraints:
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
                                min_no_of_scenarios_to_consider[risk_constraint_index] = \
                                    mid_no_of_scenarios_to_consider[risk_constraint_index]
                                print('Relaxing risk constraint', risk_constraint_index,
                                      'by updating the min. scenario bound to',
                                      min_no_of_scenarios_to_consider[risk_constraint_index])
                            else:
                                print('CVaR constraint is not satisfied')
                                all_constraints_satisfied = False
                                max_no_of_scenarios_to_consider[risk_constraint_index] = \
                                    mid_no_of_scenarios_to_consider[risk_constraint_index] - 1
                                print('Tightening risk constraint', risk_constraint_index,
                                      'by updating the max. scenario bound to',
                                      max_no_of_scenarios_to_consider[risk_constraint_index])
                                '''
                                for ind in range(len(
                                    max_no_of_scenarios_to_consider)):
                                    max_no_of_scenarios_to_consider[ind]\
                                        -= 1
                                '''

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
                                min_no_of_scenarios_to_consider[risk_constraint_index] = \
                                    mid_no_of_scenarios_to_consider[risk_constraint_index]
                                print('Updating min no. of scenarios to consider for risk'
                                      'constraint', risk_constraint_index, 'to',
                                      min_no_of_scenarios_to_consider[risk_constraint_index])
                            else:
                                print('VaR constraint not satisfied')
                                all_constraints_satisfied = False
                                max_no_of_scenarios_to_consider[risk_constraint_index] = \
                                    mid_no_of_scenarios_to_consider[risk_constraint_index] - 1
                                print('Updating max no. of scenarios to consider for risk',
                                      'constraint', risk_constraint_index, 'to',
                                      max_no_of_scenarios_to_consider[risk_constraint_index])
                                '''
                                for ind in range(len(
                                    max_no_of_scenarios_to_consider)):
                                    max_no_of_scenarios_to_consider[ind]\
                                        -= 1
                                '''
                    risk_constraint_index += 1

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
        risk_constraint_index = 0
        cvar_lower_bounds = []
        cvar_upper_bounds = []
        min_no_of_scenarios_to_consider = []
        max_no_of_scenarios_to_consider = []
        trivial_constraints = []
        for constraint in self.__query.get_constraints():
            if constraint.is_risk_constraint():
                is_satisfied = False
                if constraint.is_var_constraint():
                    if self.__validator.get_var_constraint_feasibility(
                        probabilistically_unconstrained_package,
                        constraint):
                        is_satisfied = True
                if constraint.is_cvar_constraint():
                    if self.__validator.get_cvar_constraint_feasibility(
                        probabilistically_unconstrained_package,
                        constraint):
                        is_satisfied = True
                if is_satisfied:
                    trivial_constraints.append(risk_constraint_index)
                    cvar_lower_bounds.append(constraint.get_sum_limit())
                    cvar_upper_bounds.append(constraint.get_sum_limit())
                    if constraint.is_cvar_constraint():
                        no_of_scenarios_to_consider =\
                            int(np.floor(constraint.get_percentage_of_scenarios()\
                                     *no_of_scenarios/100.0))
                    else:
                        no_of_scenarios_to_consider =\
                            int(np.floor((1 - constraint.get_probability_threshold())\
                                     *no_of_scenarios))
                    min_no_of_scenarios_to_consider.append(
                        no_of_scenarios_to_consider
                    )
                    max_no_of_scenarios_to_consider.append(
                        no_of_scenarios_to_consider
                    )
                    
                else:
                    if constraint.is_cvar_constraint():
                        cvar_threshold = \
                            self.__get_linearized_cvar_among_optimization_scenarios(
                                probabilistically_unconstrained_package,
                                constraint, no_of_scenarios
                            )
                        cvar_upper_bounds.append(cvar_threshold)
                        cvar_lower_bounds.append(
                            self.__get_expected_sum_among_optimization_scenarios(
                                probabilistically_unconstrained_package,
                                constraint.get_attribute_name()))
                        no_of_scenarios_to_consider =\
                            int(np.floor(
                                constraint.get_percentage_of_scenarios()\
                                 *no_of_scenarios/100.0))
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
                            int(np.floor(
                                cvarified_constraint.get_percentage_of_scenarios()\
                                 *no_of_scenarios/100.0))
                        min_no_of_scenarios_to_consider.append(
                            no_of_scenarios_to_consider
                        )
                        max_no_of_scenarios_to_consider.append(
                            no_of_scenarios
                        )
                risk_constraint_index += 1
        
        return cvar_upper_bounds, cvar_lower_bounds,\
            max_no_of_scenarios_to_consider,\
            min_no_of_scenarios_to_consider,\
            trivial_constraints
    

    def solve(self, can_add_scenarios = False):
        self.__metrics.start_execution()
        no_of_scenarios = self.__no_of_scenarios

        print('Setting up model')
        self.__model_setup(
            no_of_scenarios=no_of_scenarios,
            no_of_scenarios_to_consider=[],
            probabilistically_constrained=False
        )

        probabilistically_unconstrained_package = \
            self.__get_package()
        print('Probabilistically unconstrained package:',
            self.__get_package_tuple_partitions())
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
            
        print('CVaR upper bounds:', cvar_upper_bounds)
        print('CVaR lower bounds:', cvar_lower_bounds)
        print('Min No of Scenarios to consider:',
              min_no_of_scenarios_to_consider)
        print('Max no of scenarios to consider:',
              max_no_of_scenarios_to_consider)
        print('Trivial constraints:', trivial_constraints)

        init_diff = self.__l_inf(
            cvar_upper_bounds, cvar_lower_bounds)

        while self.__l_inf(cvar_upper_bounds, cvar_lower_bounds)\
            >= self.__bisection_threshold*init_diff and\
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
        return (None, 0.0, True)
    

    def display_package(self, package_dict):
        if package_dict is None:
            return
        for id in package_dict:
            print(id, ',', package_dict[id])

    
    def get_metrics(self) -> OptimizationMetrics:
        return self.__metrics