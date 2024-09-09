import gurobipy as gp
from gurobipy import GRB
import numpy as np
import psutil
from scipy import optimize

from DbInfo.DbInfo import DbInfo
from PgConnection.PgConnection import PgConnection
from SeedManager.SeedManager import SeedManager
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from StochasticPackageQuery.Query import Query
from Utils.ArcTangent import ArcTangent
from Utils.GurobiLicense import GurobiLicense
from Utils.Heap import Heap
from Utils.ObjectiveType import ObjectiveType
from Utils.RelationalOperators import RelationalOperators
from Utils.Stochasticity import Stochasticity
from Utils.TailType import TailType
from Validator.Validator import Validator
from ValueGenerator.ValueGenerator import ValueGenerator


class CVaRification:

    def __init__(self, query: Query,
                 linear_relaxation: bool,
                 dbInfo: DbInfo,
                 init_no_of_scenarios: int,
                 no_of_validation_scenarios: int,
                 approximation_bound: float,
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
        self.__grb_cvar_constraints = []

        self.__current_number_of_scenarios = 0
        self.__scenarios = dict()
        for attr in self.__get_stochastic_attributes():
            self.__scenarios[attr] = []
            for _ in range(self.__no_of_vars):
                self.__scenarios[attr].append([])

        self.__cached_coefficients = dict()
        for attr in self.__get_stochastic_attributes():
            self.__cached_coefficients[attr] = []

        self.__values = dict()
        for attr in self.__get_deterministic_attributes():
            values = \
                ValueGenerator(
                    relation=self.__query.get_relation(),
                    base_predicate=self.__query.get_base_predicate(),
                    attribute=attr).get_values()
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
                self.__query.\
                    get_objective().\
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
                self.__query.\
                    get_objective().\
                        get_attribute_name()
            )
        return attributes
    
    
    def __get_number_of_tuples(self):
        sql_query = "SELECT COUNT(*) FROM " \
            + self.__query.get_relation()
        
        if len(self.__query.get_base_predicate()) > 0:
            sql_query += " WHERE "+ \
                self.__query.get_base_predicate()
        
        sql_query += ";"
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()[0][0]

    
    def __get_upper_bound_for_vars(self):
        for constraint in \
            self.__query.get_constraints():
            if constraint.is_repeat_constraint():
                return 1 + constraint.get_repetition_limit()
        return None
    
    
    def __add_package_size_constraint_to_model(
            self, package_size_constraint):
        size_limit = \
            package_size_constraint.get_package_size_limit()
        inequality_sign = \
            package_size_constraint.get_inequality_sign()
        
        gurobi_inequality = GRB.LESS_EQUAL
        if inequality_sign == RelationalOperators.EQUALS:
            gurobi_inequality = GRB.EQUAL
        elif inequality_sign == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            gurobi_inequality = GRB.GREATER_EQUAL
        
        self.__model.addConstr(
            gp.LinExpr([1]*self.__no_of_vars, self.__vars),
            gurobi_inequality, size_limit
        )

    
    def __add_deterministic_constraint_to_model(
        self, deterministic_constraint):
        attribute = deterministic_constraint.get_attribute_name()

        inequality_sign = \
            deterministic_constraint.get_inequality_sign()
        gurobi_inequality = GRB.LESS_EQUAL
        if inequality_sign == RelationalOperators.EQUALS:
            gurobi_inequality = GRB.EQUAL
        elif inequality_sign == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            gurobi_inequality = GRB.GREATER_EQUAL
        
        sum_limit = deterministic_constraint.get_sum_limit()
        self.__model.addConstr(
            gp.LinExpr(self.__values[attribute], self.__vars),
            gurobi_inequality, sum_limit
        )

    
    def __add_expected_sum_constraint_to_model(
        self, expected_sum_constraint,
        no_of_scenarios):
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

        inequality_sign = \
            expected_sum_constraint.get_inequality_sign()
        gurobi_inequality = GRB.LESS_EQUAL
        if inequality_sign == RelationalOperators.EQUALS:
            gurobi_inequality = GRB.EQUAL
        elif inequality_sign == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            gurobi_inequality = GRB.GREATER_EQUAL
        
        sum_limit = expected_sum_constraint.get_sum_limit()

        self.__model.addConstr(
            gp.LinExpr(coefficients, self.__vars),
            gurobi_inequality, sum_limit
        )

    def __add_feasible_no_of_scenarios(self, attribute):
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

    def __add_scenarios_if_necessary(self, no_of_scenarios):
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
                            seed = SeedManager.get_next_seed(),
                            no_of_scenarios = no_of_scenarios - \
                                self.__current_number_of_scenarios
                        )
                for ind in range(len(new_scenarios)):
                    for value in new_scenarios[ind]:
                        self.__scenarios[attr][ind].append(value)
            self.__current_number_of_scenarios = no_of_scenarios

    
    def __get_values_to_consider(self, cvar_constraint,
                                no_of_scenarios):
        return int(np.floor(
            (cvar_constraint.get_percentage_of_scenarios()\
                *no_of_scenarios)/100))
    
    def __get_cvar_constraint_coefficients(
        self, cvar_constraint: CVaRConstraint,
        no_of_scenarios
    ):
        attr = cvar_constraint.get_attribute_name()
        values_to_consider = self.__get_values_to_consider(
            cvar_constraint, no_of_scenarios
        )
            
        tuple_wise_heaps = []
        if cvar_constraint.get_tail_type() == TailType.HIGHEST:
            for _ in range(self.__no_of_vars):
                tuple_wise_heaps.append(
                    Heap(is_max_heap=False))
        else:
            for _ in range(self.__no_of_vars):
                tuple_wise_heaps.append(
                    Heap(is_max_heap=True))
        for scenario_no in range(no_of_scenarios):

            scenario_index = scenario_no % \
                self.__feasible_no_of_scenarios_to_store
            
            if scenario_index == 0 and scenario_no != 0:
                self.__add_feasible_no_of_scenarios(attr)

            for var in range(self.__no_of_vars):
                scenario_value = self.__scenarios[attr][var][
                    scenario_index]
                tuple_wise_heaps[var].push(scenario_value)
                if tuple_wise_heaps[var].size() > values_to_consider:
                    value = tuple_wise_heaps[var].pop()
                
        self.__cached_coefficients[attr] = [tuple_wise_heaps[var].sum()/\
                values_to_consider for var in range(self.__no_of_vars)]


    def __add_cvar_constraint_to_model(
        self, cvar_constraint: CVaRConstraint,
        no_of_scenarios, cvar_threshold
    ):
        attr = cvar_constraint.get_attribute_name()
        
        if len(self.__cached_coefficients[attr]) == 0:
            self.__get_cvar_constraint_coefficients(
                cvar_constraint, no_of_scenarios
            )

        inequality_sign = \
            cvar_constraint.get_inequality_sign()
        gurobi_inequality = GRB.LESS_EQUAL
        if inequality_sign == RelationalOperators.EQUALS:
            gurobi_inequality = GRB.EQUAL
        elif inequality_sign == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            gurobi_inequality = GRB.GREATER_EQUAL        

        grb_cvar_constraint = self.__model.addConstr(
            gp.LinExpr(
                self.__cached_coefficients[attr], self.__vars),
            gurobi_inequality, cvar_threshold
        )
        self.__grb_cvar_constraints.append(grb_cvar_constraint)


    def __add_constraints_to_model(
        self, no_of_scenarios, probabilistically_constrained,
        cvar_thresholds, trivial_constraints):
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
                                CVaRConstraint()
                            cvarified_constraint.\
                                set_percentage_of_scenarios(
                                    (1 - \
                                     constraint.\
                                        get_probability_threshold()
                                    )*100
                                )
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
                                cvar_thresholds[risk_constraint_index]
                            )
                            
                        self.__add_cvar_constraint_to_model(
                            cvarified_constraint, no_of_scenarios,
                            cvar_thresholds[risk_constraint_index]
                        )
                    risk_constraint_index += 1

    
    def __add_objective_to_model(
        self, objective, no_of_scenarios):
        attr = objective.get_attribute_name()
        coefficients = []
        
        if objective.get_stochasticity() == \
            Stochasticity.DETERMINISTIC:
            coefficients = self.__values[attr]
        else:
            if no_of_scenarios <= \
                self.__feasible_no_of_scenarios_to_store:
                self.__add_scenarios_if_necessary(
                    no_of_scenarios)
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

    
    def __add_variables_to_model(self):
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
                        lb = 0,
                        vtype = type
                    )
                )
    
    
    def __model_setup(
            self, no_of_scenarios: int,
            probabilistically_constrained = True,
            cvar_thresholds = [],
            trivial_constraints = [],
         ):
        
        self.__model = gp.Model(
            env=self.__gurobi_env)
        self.__add_variables_to_model()
        self.__grb_cvar_constraints = []
        self.__add_constraints_to_model(
            no_of_scenarios, probabilistically_constrained,
            cvar_thresholds, trivial_constraints)
        self.__add_objective_to_model(
            self.__query.get_objective(), no_of_scenarios)

    
    def __l_inf(self, l1: list[float],
                l2: list[float]) -> float:
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


    def __get_package(self):
        self.__model.optimize()
        package_dict = {}
        idx = 0
        try:
            for var in self.__vars:
                if var.x > 0:
                    package_dict[self.__ids[idx]] = var.x
                idx += 1
        except AttributeError:
            return None
        return package_dict


    def __cvar_solve(self, no_of_scenarios: int,
                    cvar_upper_bounds: list[float],
                    cvar_lower_bounds: list[float],
                    trivial_constraints: list[int],
                    objective_upper_bound: float):
        
        print('Init CVaR Upper Bounds:', cvar_upper_bounds)
        print('Init CVaR lower bounds:', cvar_lower_bounds)
        is_model_setup = False
        while self.__l_inf(
            cvar_upper_bounds, cvar_lower_bounds) >= \
            self.__bisection_threshold:

            cvar_mid_thresholds = []
            for ind in range(len(cvar_lower_bounds)):
                cvar_mid_thresholds.append(
                    (cvar_lower_bounds[ind] +\
                     cvar_upper_bounds[ind]) / 2.0
                )
            
            print('CVaR mid thresholds:', cvar_mid_thresholds)
            if is_model_setup:
                cvar_constraint_index = 0
                for threshold in cvar_mid_thresholds:
                    self.__grb_cvar_constraints[
                        cvar_constraint_index].rhs = threshold
                    cvar_constraint_index += 1
            else:
                self.__model_setup(
                    no_of_scenarios=no_of_scenarios,
                    probabilistically_constrained = True,
                    cvar_thresholds=cvar_mid_thresholds,
                    trivial_constraints=trivial_constraints)
                is_model_setup = True
            
            package = self.__get_package()
            print('Package:', package)
            
            if self.__validator.is_package_validation_feasible(
                package):
                cvar_lower_bounds = cvar_mid_thresholds
                print('CVaR lower bounds become:', cvar_lower_bounds)
                if self.__validator.is_package_1_pm_epsilon_approximate(
                    package, self.__approximation_bound,
                    objective_upper_bound):
                    return (package, objective_upper_bound)
            else:
                print('Infeasible package:', package)
                objective_value = self.__validator.\
                    get_validation_objective_value(
                        package)
                print('Objective Value:', objective_value)
                if self.__query.get_objective().get_objective_type() ==\
                    ObjectiveType.MAXIMIZATION:
                    if objective_value < objective_upper_bound:
                        objective_upper_bound = objective_value
                elif objective_value > objective_upper_bound:
                    objective_upper_bound = objective_value
                
                cvar_upper_bounds = cvar_mid_thresholds
                print('CVaR Upper Bounds become', cvar_upper_bounds)
        
        return (package, objective_upper_bound)


    def __get_linearized_cvar_among_optimization_scenarios(
        self, package_dict, cvar_constraint, no_of_scenarios):
        attr = cvar_constraint.get_attribute_name()
        if len(self.__cached_coefficients[attr]) == 0:
            self.__get_cvar_constraint_coefficients(
                cvar_constraint, no_of_scenarios)
        
        linearized_cvar = 0
        idx = 0
        for id in self.__ids:
            if id in package_dict:
                linearized_cvar += \
                    self.__cached_coefficients[attr][idx] * \
                    package_dict[id]
            idx += 1
        return linearized_cvar/self.__get_values_to_consider(
            cvar_constraint, no_of_scenarios
        )


    def get_bounds_for_risk_constraints(
            self, no_of_scenarios,
            probabilistically_unconstrained_package):
        self.__add_scenarios_if_necessary(no_of_scenarios)
        risk_constraint_index = 0
        cvar_lower_bounds = []
        cvar_upper_bounds = []
        trivial_constraints = []
        for constraint in self.__query.get_constraints():
            if constraint.is_risk_constraint():
                is_satisfied = False
                if constraint.is_var_constraint():
                    if self.__validator.get_var_constraint_feasibility(
                        probabilistically_unconstrained_package, constraint):
                        is_satisfied = True
                if constraint.is_cvar_constraint():
                    if self.__validator.get_cvar_constraint_feasibility(
                        probabilistically_unconstrained_package, constraint):
                        is_satisfied = True
                if is_satisfied:
                    trivial_constraints.append(risk_constraint_index)
                    cvar_lower_bounds.append(constraint.get_sum_limit())
                    cvar_upper_bounds.append(constraint.get_sum_limit())
                else:
                    if constraint.is_cvar_constraint():
                        cvar_threshold = \
                            self.__get_linearized_cvar_among_optimization_scenarios(
                                probabilistically_unconstrained_package, constraint,
                                no_of_scenarios
                            )
                        cvar_upper_bounds.append(cvar_threshold)
                        cvar_lower_bounds.append(constraint.get_sum_limit())
                    else:
                        cvarified_constraint = \
                            CVaRConstraint()
                        cvarified_constraint.set_percentage_of_scenarios(
                            (1 - constraint.get_probability_threshold())*100)
                            
                        if constraint.get_inequality_sign() == \
                            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                            cvarified_constraint.set_tail_type('l')
                        else:
                            cvarified_constraint.set_tail_type('h')
                    
                        if constraint.get_inequality_sign() \
                            == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                            cvarified_constraint.set_inequality_sign('>')
                        else:    
                            cvarified_constraint.set_inequality_sign('<')
                    
                        cvarified_constraint.set_attribute_name(
                            constraint.get_attribute_name()
                        )
                        cvarified_constraint.set_sum_limit(
                            constraint.get_sum_limit()
                        )
                        cvar_threshold = \
                            self.__get_linearized_cvar_among_optimization_scenarios(
                                probabilistically_unconstrained_package, cvarified_constraint,
                                no_of_scenarios
                            )

                        cvar_upper_bounds.append(cvar_threshold)
                        cvar_lower_bounds.append(constraint.get_sum_limit())
                risk_constraint_index += 1

        return cvar_upper_bounds, cvar_lower_bounds, trivial_constraints


    def solve(self):
        no_of_scenarios = self.__init_no_of_scenarios

        self.__model_setup(
            no_of_scenarios,
            probabilistically_constrained=False
        )

        probabilistically_unconstrained_package = \
            self.__get_package()
        print('Probabilistically unconstrained package',
              probabilistically_unconstrained_package)
        
        if probabilistically_unconstrained_package is None:
            return None
        
        objective_upper_bound = \
            self.__validator.get_validation_objective_value(
                probabilistically_unconstrained_package
            )

        if self.__validator.is_package_validation_feasible(
            probabilistically_unconstrained_package):
            return (probabilistically_unconstrained_package,
                    objective_upper_bound)
        
        while no_of_scenarios <= self.__no_of_validation_scenarios:
            print('Trying with', no_of_scenarios, 'scenarios')

            cvar_upper_bounds, cvar_lower_bounds, trivial_constraints =\
                self.get_bounds_for_risk_constraints(
                    no_of_scenarios,
                    probabilistically_unconstrained_package)
            
            print('CVaR Lower Bounds:', cvar_lower_bounds)
            print('CVaR Upper Bounds:', cvar_upper_bounds)

            package, objective_upper_bound = \
                self.__cvar_solve(
                    no_of_scenarios, cvar_upper_bounds,
                    cvar_lower_bounds, trivial_constraints,
                    objective_upper_bound
                )

            if self.__validator.is_package_validation_feasible(package):
                if self.__validator.is_package_1_pm_epsilon_approximate(
                    package, self.__approximation_bound,
                    objective_upper_bound):
                    return package
                else:
                    print('Package feasible but not optimal enough')
            else:
                print('Package infeasible')
            
            no_of_scenarios *= 2
            for attr in self.__get_stochastic_attributes():
                self.__cached_coefficients[attr] = []
        
        return None
    
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

    def __del__(self):
        self.__model.close()
        self.__gurobi_env.close()