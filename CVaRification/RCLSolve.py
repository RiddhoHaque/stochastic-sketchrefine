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
from Validator.Validator import Validator
from ValueGenerator.ValueGenerator import ValueGenerator


class RCLSolve:

    def __init__(self, query: Query,
                 linear_relaxation: bool,
                 dbInfo: DbInfo,
                 init_no_of_scenarios: int,
                 no_of_validation_scenarios: int,
                 approximation_bound: float,
                 sampling_tolerance: float,
                 bisection_threshold: float):
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
        self.__init_no_of_scenarios = \
            init_no_of_scenarios
        
        self.__no_of_validation_scenarios = \
            no_of_validation_scenarios
        self.__validator = Validator(
            self.__query, dbInfo,
            self.__no_of_validation_scenarios
        )
        self.__approximation_bound = \
            approximation_bound
        self.__sampling_tolerance = \
            sampling_tolerance
        self.__bisection_threshold = \
            bisection_threshold
        
        self.__no_of_vars = \
            self.__get_number_of_tuples()
        self.__feasible_no_of_scenarios_to_store = \
            int(np.floor(
                (0.40*psutil.virtual_memory().available)/\
                (64*self.__no_of_vars)
            ))

        self.__vars = []
        
        self.__risk_constraints = []
        self.__risk_to_lcvar_constraint_mapping = dict()


        self.__current_number_of_scenarios = 0
        
        self.__scenarios = dict()

        for attr in self.__get_stochastic_attributes():
            self.__scenarios[attr] = []
            for _ in range(self.__no_of_vars):
                self.__scenarios[attr].append([])
        
        self.__values = dict()
        for attr in self.__get_deterministic_attributes():
            values = \
                ValueGenerator(
                    relation=self.__query.get_relation(),
                    base_predicate=self.__query.get_base_predicate(),
                    attribute=attr
                ).get_values()
            self.__values[attr] = []
            for value in values:
                self.__values[attr].append(value[0])

        self.__dbInfo = dbInfo
        self.__ids = []
        ids = ValueGenerator(
                relation=self.__query.get_relation(),
                base_predicate=self.__query.get_base_predicate(),
                attribute='id'
            ).get_values()
        for id in ids:
            self.__ids.append(id[0])
        self.__metrics = OptimizationMetrics(
            'RCLSolve', self.__is_linear_relaxation
        )

    
    def __get_stochastic_attributes(self):
        attributes = set()
        for constraint in self.__query.get_constraints():
            if constraint.is_expected_sum_constraint():
                attributes.add(
                    constraint.get_attribute_name())
            if constraint.is_risk_constraint():
                attributes.add(
                    constraint.get_attribute_name())
        if self.__query.get_objective().get_stochasticity() \
            == Stochasticity.STOCHASTIC:
            attributes.add(
                self.__query.get_objective().\
                    get_attribute_name()
            )
        return attributes


    def __get_deterministic_attributes(self):
        attributes = set()
        for constraint in self.__query.get_constraints():
            if constraint.is_deterministic_constraint():
                attributes.add(
                    constraint.get_attribute_name())
        
        if self.__query.get_objective().get_stochasticity() \
            == Stochasticity.DETERMINISTIC:
            attributes.add(
                self.__query.get_objective().\
                    get_attribute_name()
            )
        return attributes
    

    def __get_number_of_tuples(self):
        sql_query = "SELECT COUNT(*) FROM " \
            + self.__query.get_relation()
        
        if len(self.__query.get_base_predicate()) > 0:
            sql_query += " WHERE " + \
                self.__query.get_base_predicate()
        
        sql_query += ";"
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()[0][0]
    

    def __get_upper_bound_for_vars(self) -> int:
        for constraint in self.__query.get_constraints():
            if constraint.is_repeat_constraint():
                return 1 + constraint.get_repetition_limit()
        return None
    
    
    def __add_variables_to_model(self) -> None:
        max_repetition = \
            self.__get_upper_bound_for_vars()
        type = GRB.INTEGER
        if self.__is_linear_relaxation:
            type = GRB.CONTINUOUS
        self.__vars = []
        if max_repetition is not None:
            for _ in range(self.__no_of_vars):
                self.__vars.append(
                    self.__model.addVar(
                        lb=0,
                        ub=max_repetition,
                        vtype=type
                    )
                )
        else:
            for _ in range(self.__no_of_vars):
                self.__vars.append(
                    self.__model.addVar(
                        lb=0,
                        vtype=type
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


    def __add_feasible_no_of_scenarios(self, attribute: str):
        self.__scenarios[attribute] = \
            self.__dbInfo.get_variable_generator_function(
                attribute)(
                    self.__query.get_relation(),
                    self.__query.get_base_predicate()
                ).generate_scenarios(
                    seed = SeedManager.get_next_seed(),
                    no_of_scenarios = \
                        self.__feasible_no_of_scenarios_to_store
                )
        
    
    def __add_all_scenarios_if_possible(self, no_of_scenarios):
        if no_of_scenarios > self.__feasible_no_of_scenarios_to_store:
            return
        
        if self.__current_number_of_scenarios < \
            no_of_scenarios:
            
            for attr in self.__scenarios:
                
                new_scenarios = \
                    self.__dbInfo.get_variable_generator_function(
                        attr)(
                            self.__query.get_relation(),
                            self.__query.get_base_predicate()
                        ).generate_scenarios(
                            seed = Hyperparameters.VALIDATION_SEED,
                            no_of_scenarios = no_of_scenarios - \
                                self.__current_number_of_scenarios
                        )
                
                for ind in range(len(new_scenarios)):
                    for value in new_scenarios[ind]:
                        self.__scenarios[attr][ind].append(value)
            
            self.__current_number_of_scenarios = no_of_scenarios


    def __add_expected_sum_constraint_to_model(
        self, expected_sum_constraint: ExpectedSumConstraint,
        no_of_scenarios: int
    ):
        attr = expected_sum_constraint.get_attribute_name()
        coefficients = []

        if no_of_scenarios <= \
            self.__feasible_no_of_scenarios_to_store:
            for idx in range(self.__no_of_vars):
                coefficients.append(
                    np.average(
                        self.__scenarios[attr][idx]
                    )
                )
        
        else:
            
            total_scenarios = 0
            coefficient_set = [[] for _ in range(
                self.__no_of_vars)]
            
            while total_scenarios < no_of_scenarios:
                self.__add_feasible_no_of_scenarios(attr)
                total_scenarios += \
                    self.__feasible_no_of_scenarios_to_store
                for idx in range(self.__no_of_vars):
                    coefficient_set[idx].append(
                        np.average(
                            self.__scenarios[attr][idx]
                        )
                    )

            for idx in range(self.__no_of_vars):
                coefficients.append(
                    np.average(coefficient_set[idx])
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
                *no_of_scenarios)/100
        ))


    def __get_cvar_constraint_coefficients(
        self, cvar_constraint: CVaRConstraint,
        no_of_scenarios_to_consider: int,
        total_no_of_scenarios: int
    ):
        attr = cvar_constraint.get_attribute_name()

        tuple_wise_heaps = []
        is_max_heap = True
        if cvar_constraint.get_tail_type() == TailType.HIGHEST:
            is_max_heap = False
        
        for _ in range(self.__no_of_vars):
            tuple_wise_heaps.append(
                Heap(is_max_heap))
        
        for scenario_no in range(total_no_of_scenarios):

            scenario_index = scenario_no % \
                self.__feasible_no_of_scenarios_to_store
            
            if scenario_index == 0 and scenario_no != 0:
                self.__add_feasible_no_of_scenarios(attr)
            
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
        cvarified_constraint = \
            CVaRConstraint()
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
        self.__add_all_scenarios_if_possible(
            no_of_scenarios
        )
        risk_constraint_index = 0
        for constraint in self.__query.get_constraints():
            if constraint.is_package_size_constraint():
                self.__add_package_size_constraint_to_model(
                    constraint
                )
            if constraint.is_deterministic_constraint():
                self.__add_deterministic_constraint_to_model(
                    constraint
                )
            if constraint.is_expected_sum_constraint():
                if probabilistically_constrained:
                    self.__add_expected_sum_constraint_to_model(
                        constraint, no_of_scenarios
                    )
            if constraint.is_risk_constraint():
                if probabilistically_constrained:
                    if risk_constraint_index not in \
                        trivial_constraints:
                        if constraint.is_cvar_constraint():
                            cvarified_constraint = \
                                constraint
                        else:
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint=constraint,
                                    cvar_threshold=cvar_thresholds[
                                        risk_constraint_index]
                                )
                        
                        self.__add_lcvar_constraint_to_model(
                            risk_constraint=constraint,
                            cvarified_constraint=cvarified_constraint,
                            no_of_scenarios=no_of_scenarios,
                            no_of_scenarios_to_consider=\
                                no_of_scenarios_to_consider[
                                    risk_constraint_index
                                ]
                        )
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
            if no_of_scenarios <= \
                self.__feasible_no_of_scenarios_to_store:
                for idx in range(self.__no_of_vars):
                    coefficients.append(
                        np.average(
                            self.__scenarios[attr][idx]
                        )
                    )
            else:
                total_scenarios = 0
                coefficient_set = [[] for _ in \
                                    range(self.__no_of_vars)]
                while total_scenarios < no_of_scenarios:
                    self.__add_feasible_no_of_scenarios(attr)
                    total_scenarios += \
                        self.__feasible_no_of_scenarios_to_store
                    for idx in range(self.__no_of_vars):
                        coefficient_set[idx].append(
                            np.average(
                                self.__scenarios[attr][idx]
                            )
                        )
                
                for idx in range(self.__no_of_vars):
                    coefficients.append(
                        np.average(coefficient_set[idx])
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
        self.__add_variables_to_model()
        self.__add_constraints_to_model(
            no_of_scenarios,
            no_of_scenarios_to_consider,
            probabilistically_constrained,
            cvar_lower_bounds, trivial_constraints)
        self.__add_objective_to_model(
            self.__query.get_objective(),
            no_of_scenarios)
        

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
        return package_dict
    

    def __get_package_with_indices(self):
        package_dict = {}
        idx = 0
        try:
            for var in self.__vars:
                if var.x > 0:
                    package_dict[
                        idx] = var.x
                idx += 1
        except AttributeError:
            return None
        return package_dict
    

    def __get_objective_value_among_optimization_scenarios(
        self, package_with_indices, no_of_scenarios
    ) -> float:
        if package_with_indices is None:
            return 0
        attr = self.__query.get_objective().get_attribute_name()
        if no_of_scenarios < \
            self.__feasible_no_of_scenarios_to_store:
            sum = 0
            for idx in package_with_indices:
                sum += np.average(self.__scenarios[attr][idx]) * \
                    package_with_indices[idx]
                #print('Index in package:', idx)
                #print('Avg. Value:', np.average(self.__scenarios[attr][idx]))
            return sum
        
        sum = 0
        total_scenarios = 0
        while total_scenarios < no_of_scenarios:
            self.__add_feasible_no_of_scenarios(attr)
            total_scenarios += no_of_scenarios
            for idx in package_with_indices:
                for value in self.__scenarios[attr][idx]:
                    sum += value * self.__vars[idx].x
        return sum / total_scenarios
    

    def __is_objective_value_relative_diff_high(
        self, package, package_with_indices,
        no_of_scenarios
    ) -> bool:
        if package is None:
            return False, 0
        objective_value_optimization_scenarios =\
            self.__get_objective_value_among_optimization_scenarios(
                package_with_indices, no_of_scenarios
            )
        print('Obj value optimization:',
              objective_value_optimization_scenarios)
        objective_value_validation_scenarios =\
            self.__validator.get_validation_objective_value(
                package
            )
        print('Obj value validation:',
              objective_value_validation_scenarios)
        diff = objective_value_optimization_scenarios - \
                    objective_value_validation_scenarios
        
        if self.__query.get_objective().get_objective_type() ==\
            ObjectiveType.MINIMIZATION:
            diff *= -1

        rel_diff = diff / (objective_value_validation_scenarios
            + 0.00001)
        
        print('Relative difference:', rel_diff)
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
            print('Objective Value is', objective_value)
            print('Upper bound is', objective_upper_bound)
            print('Epsilon is', self.__approximation_bound)
            print('Objective Value needs to be at least',
                  (1 - self.__approximation_bound) *
                    objective_upper_bound)
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
        scenario_scores = []
        if no_of_scenarios < \
            self.__feasible_no_of_scenarios_to_store:
            for scenario_no in range(no_of_scenarios):
                scenario_score = 0
                
                for idx in package_with_indices:
                    value = \
                        self.__scenarios[attr][idx][
                            scenario_no]
                    scenario_score += value *\
                        package_with_indices[idx]
                
                scenario_scores.append(scenario_score)
            scenario_scores.sort()
            return scenario_scores
        
        total_scenarios = 0
        while total_scenarios < \
            self.__feasible_no_of_scenarios_to_store:
            
            self.__add_feasible_no_of_scenarios(attr)
            total_scenarios += \
                self.__feasible_no_of_scenarios_to_store
            
            for scenario_no in range(
                self.__feasible_no_of_scenarios_to_store):
                scenario_score = 0

                for idx in package_with_indices:
                    value = \
                        self.__scenarios[attr][idx][
                            scenario_no
                        ]
                    scenario_score += value * \
                        package_with_indices[idx]
                
                scenario_scores.append(scenario_score)
        
        scenario_scores.sort()
        return scenario_scores
        
    
    def __get_cvar_among_optimization_scenarios(
        self, package_with_indices,
        no_of_scenarios: int,
        constraint: CVaRConstraint
    ) -> float:
        scenario_scores = \
            self.__get_scenario_scores_ascending(
                package_with_indices, no_of_scenarios,
                constraint
            )
        if constraint.get_tail_type() == TailType.HIGHEST:
            scenario_scores.reverse()
        scenarios_to_consider = \
            np.floor((constraint.get_percentage_of_scenarios()\
                      *no_of_scenarios))/100
        return np.average(
            scenario_scores[0:scenarios_to_consider])
    

    def __get_var_among_optimization_scenarios(
        self, package_with_indices,
        no_of_scenarios: int,
        constraint: VaRConstraint
    ) -> float:
        scenario_scores = \
            self.__get_scenario_scores_ascending(
                package_with_indices, no_of_scenarios,
                constraint
            )
        scenarios_to_consider = \
            int(np.floor((constraint.get_probability_threshold()*\
                      no_of_scenarios)))
        return scenario_scores[scenarios_to_consider]
    

    def __is_var_relative_difference_high(
        self, package_with_indices,
        package, no_of_scenarios,
        constraint
    ) -> bool:
        if package is None:
            return False, constraint.get_sum_limit()
        
        var_optimization = \
            self.__get_var_among_optimization_scenarios(
                package_with_indices, no_of_scenarios,
                constraint
            )

        var_validation = \
            self.__validator.get_var_among_validation_scenarios(
                package, constraint
            )
        

        if constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            rel_diff = var_optimization - var_validation
        else:
            rel_diff= var_validation - var_optimization
        rel_diff /= (var_validation + 0.000001)

        if rel_diff > self.__sampling_tolerance:
            return True, var_validation
        return False, var_validation
    

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

    
    def __is_cvar_relative_difference_high(
        self, package_with_indices,
        package, no_of_scenarios,
        constraint
    ) -> bool:
        if package is None:
            return False, constraint.get_sum_limit()
        
        cvar_optimization = \
            self.__get_cvar_among_optimization_scenarios(
                package_with_indices, no_of_scenarios,
                constraint
            )
        cvar_validation = \
            self.__validator.get_cvar_constraint_satisfaction(
                package, constraint
            )
        
        if constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            rel_diff = cvar_optimization - cvar_validation
        else:
            rel_diff= cvar_validation - cvar_optimization
        
        rel_diff /= (cvar_validation + 0.0000001)
        
        if rel_diff > self.__sampling_tolerance:
            return True, cvar_validation
        return False, cvar_validation
    

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
                self.__get_package_with_indices()
            print('Packages with indices:', package_with_indices)
            
            unacceptable_diff, validation_objective_value = \
                self.__is_objective_value_relative_diff_high(
                    package, package_with_indices,
                    no_of_scenarios
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
                            unacceptable_diff, cvar_validation = \
                                self.__is_cvar_relative_difference_high(
                                    package_with_indices, package,
                                    no_of_scenarios, constraint
                                )
                            if unacceptable_diff and can_add_scenarios:
                                print('Unacceptable diff for cvar')
                                result = CVaRificationSearchResults()
                                result.set_needs_more_scenarios(True)
                                return result
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
                            unacceptable_diff, var_validation = \
                                self.__is_var_relative_difference_high(
                                    package_with_indices, package,
                                    no_of_scenarios, constraint
                                )
                            if unacceptable_diff and can_add_scenarios:
                                print('Unacceptable diff for var')
                                result = CVaRificationSearchResults()
                                result.set_needs_more_scenarios(True)
                                return result
                            if self.__is_var_constraint_satisfied(
                                constraint, var_validation
                            ):
                                print('VaR constraint satisfied')
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
                        result.set_objective_value(
                            validation_objective_value)
                        return result
                
                if all_constraints_violated:
                    print('All constraints were violated')
                    if self.__query.get_objective().get_objective_type() ==\
                        ObjectiveType.MAXIMIZATION:

                        if validation_objective_value < objective_upper_bound:
                            print('Updating objective upper bound to',
                                  validation_objective_value)
                            #objective_upper_bound = validation_objective_value
                    
                    
                    else:
                        if validation_objective_value > objective_upper_bound:
                            print('Updating objective upper bound to',
                                  validation_objective_value)
                            #objective_upper_bound = validation_objective_value
            
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
                risk_constraint_index = 0
                for constraint in self.__query.get_constraints():
                    if constraint.is_risk_constraint():
                        if risk_constraint_index not in trivial_constraints:
                            coefficients = \
                                self.__get_cvar_constraint_coefficients(
                                    self.__get_cvarified_constraint(
                                        constraint, cvar_thresholds[
                                            risk_constraint_index]),
                                    mid_no_of_scenarios_to_consider[
                                        risk_constraint_index],
                                    no_of_scenarios)
                        
                            for _ in range(len(self.__vars)):
                                self.__model.chgCoeff(
                                    self.__risk_to_lcvar_constraint_mapping[
                                        constraint],
                                    self.__vars[_], coefficients[_]
                                )                    
                        risk_constraint_index += 1
            
            package = self.__get_package()
            if package is None:
                min_no_of_scenarios_to_consider = \
                    mid_no_of_scenarios_to_consider
                continue

            package_with_indices = \
                self.__get_package_with_indices()
            
            unacceptable_diff, validation_objective_value = \
                self.__is_objective_value_relative_diff_high(
                    package, package_with_indices,
                    no_of_scenarios
                )
            
            if unacceptable_diff and can_add_scenarios:
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
                            unacceptable_diff, cvar_validation = \
                                self.__is_cvar_relative_difference_high(
                                    package_with_indices, package,
                                    no_of_scenarios, constraint
                                )
                            if unacceptable_diff and can_add_scenarios:
                                result = CVaRificationSearchResults()
                                result.set_needs_more_scenarios(True)
                                return result
                            
                            if self.__is_cvar_constraint_satisfied(
                                constraint, cvar_validation
                            ):
                                all_constraints_violated = False
                                min_no_of_scenarios_to_consider = \
                                    mid_no_of_scenarios_to_consider
                            else:
                                all_constraints_satisfied = False
                                max_no_of_scenarios_to_consider = \
                                    mid_no_of_scenarios_to_consider
                                
                                for ind in range(len(
                                    max_no_of_scenarios_to_consider)):
                                    max_no_of_scenarios_to_consider[ind]\
                                        -= 1

                        if constraint.is_var_constraint():
                            unacceptable_diff, var_validation = \
                                self.__is_var_relative_difference_high(
                                    package_with_indices, package,
                                    no_of_scenarios, constraint
                                )
                            if unacceptable_diff and can_add_scenarios:
                                result = CVaRificationSearchResults()
                                result.set_needs_more_scenarios(True)
                                return result
                            
                            if self.__is_var_constraint_satisfied(
                                constraint, var_validation
                            ):
                                all_constraints_violated = False
                                min_no_of_scenarios_to_consider = \
                                    mid_no_of_scenarios_to_consider
                            else:
                                all_constraints_satisfied = False
                                max_no_of_scenarios_to_consider = \
                                    mid_no_of_scenarios_to_consider
                                
                                for ind in range(len(
                                    max_no_of_scenarios_to_consider)):
                                    max_no_of_scenarios_to_consider[ind]\
                                        -= 1
                    risk_constraint_index += 1

                if all_constraints_satisfied:
                    if self.__is_objective_value_enough(
                        validation_objective_value, objective_upper_bound
                    ):
                        result = CVaRificationSearchResults()
                        result.set_found_appropriate_package(True)
                        result.set_package(package)
                        result.set_objective_value(
                            validation_objective_value)
                        return result
                
                if all_constraints_violated:
                    if self.__query.get_objective().get_objective_type() == \
                        Objective.MAXIMIZATION:
                        
                        if validation_objective_value < objective_upper_bound:
                            ...
                            #objective_upper_bound = validation_objective_value
                        
                    else:
                        if validation_objective_value > objective_upper_bound:
                            ...
                            #objective_upper_bound = validation_objective_value

        result = CVaRificationSearchResults()
        result.set_objective_upper_bound(objective_upper_bound)
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
    

    def __get_bounds_for_risk_constraints(
        self, no_of_scenarios,
        probabilistically_unconstrained_package
    ):
        self.__add_all_scenarios_if_possible(
            no_of_scenarios)
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
                    no_of_scenarios_to_consider =\
                        np.floor(constraint.get_percentage_of_scenarios()\
                                 *no_of_scenarios/100.0)
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
                        cvar_lower_bounds.append(constraint.get_sum_limit())
                        no_of_scenarios_to_consider =\
                            np.floor(
                                constraint.get_percentage_of_scenarios()\
                                 *no_of_scenarios/100.0)
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
                        cvar_lower_bounds.append(constraint.get_sum_limit())
                        no_of_scenarios_to_consider =\
                            np.floor(
                                cvarified_constraint.get_percentage_of_scenarios()\
                                 *no_of_scenarios/100.0)
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
    

    def solve(self, can_add_scenarios = True):
        self.__metrics.start_execution()
        no_of_scenarios = self.__init_no_of_scenarios
        unacceptable_diff = True

        while unacceptable_diff:
            self.__model_setup(
                no_of_scenarios=no_of_scenarios,
                no_of_scenarios_to_consider=[],
                probabilistically_constrained=False
            )

            probabilistically_unconstrained_package = \
                self.__get_package()
            print('Probabilistically unconstrained package:',
                probabilistically_unconstrained_package)
            if probabilistically_unconstrained_package is None:
                self.__metrics.end_execution(0)
                return None
        
            probabilistically_unconstrained_package_with_indices = \
                self.__get_package_with_indices()

            unacceptable_diff, validation_objective_value = \
                self.__is_objective_value_relative_diff_high(
                    probabilistically_unconstrained_package,
                    probabilistically_unconstrained_package_with_indices,
                    no_of_scenarios
                )
            
            if not can_add_scenarios or not unacceptable_diff:
                objective_upper_bound = validation_objective_value
                break

            no_of_scenarios *= 2
        
        print('Objective value upper bound:',
              objective_upper_bound)

        if self.__validator.is_package_validation_feasible(
            probabilistically_unconstrained_package):
            print('Probabilistically unconstrained package'
                  'is validation feasible')
            self.__metrics.end_execution(objective_upper_bound)
            return (probabilistically_unconstrained_package,
                    objective_upper_bound)

        while no_of_scenarios <= self.__no_of_validation_scenarios:

            cvar_upper_bounds, cvar_lower_bounds,\
            max_no_of_scenarios_to_consider,\
            min_no_of_scenarios_to_consider,\
            trivial_constraints = \
                self.__get_bounds_for_risk_constraints(
                    no_of_scenarios,
                    probabilistically_unconstrained_package,
                )

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
                
                is_model_setup = True

                if coefficient_search_result.\
                    get_needs_more_scenarios():
                    break
                if coefficient_search_result.\
                    get_found_appropriate_package():
                    self.__metrics.end_execution(
                        coefficient_search_result.\
                            get_objective_value()
                    )
                    return (
                        coefficient_search_result.get_package(),
                        coefficient_search_result.get_objective_value()
                    )
                
                min_no_of_scenarios_to_consider = \
                    coefficient_search_result.get_scenarios_to_consider()
                objective_upper_bound = \
                    coefficient_search_result.get_objective_upper_bound()
                
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

                if threshold_search_result.\
                    get_needs_more_scenarios():
                    break
                if threshold_search_result.\
                    get_found_appropriate_package():
                    self.__metrics.end_execution(
                        threshold_search_result.\
                            get_objective_value()
                    )
                    return (
                        threshold_search_result.get_package(),
                        threshold_search_result.get_objective_value()
                    )
                cvar_upper_bounds = \
                    threshold_search_result.get_cvar_thresholds()
                objective_upper_bound = \
                    threshold_search_result.get_objective_upper_bound()
                
                for ind in range(len(min_no_of_scenarios_to_consider)):
                    if min_no_of_scenarios_to_consider[ind] <\
                        max_no_of_scenarios_to_consider[ind]:
                        min_no_of_scenarios_to_consider[ind] += 1
            
            no_of_scenarios *= 2
            if no_of_scenarios >= self.__no_of_validation_scenarios:
                no_of_scenarios = self.__no_of_validation_scenarios
        self.__metrics.end_execution(0)
        return (None, 0.0)
    

    def __get_attributes(self, id):
        sql_query = "SELECT " + \
            self.__query.get_projected_attributes() + \
            " FROM " + self.__query.get_relation() + \
            " WHERE id=" + str(id) + \
            " ORDER BY id;"
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()[0]
    

    def display_package(self, package_dict):
        if package_dict is None:
            return
        for id in package_dict:
            attr = self.__get_attributes(id)
            print(attr, ',', package_dict[id])

    def get_metrics(self) -> OptimizationMetrics:
        return self.__metrics