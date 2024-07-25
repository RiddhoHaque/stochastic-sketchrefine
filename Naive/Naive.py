import gurobipy as gp
from gurobipy import GRB
import numpy as np
import psutil
from DbInfo.DbInfo import DbInfo
from PgConnection.PgConnection import PgConnection
from SeedManager.SeedManager import SeedManager
from StochasticPackageQuery.Query import Query
from Utils.GurobiLicense import GurobiLicense
from Utils.ObjectiveType import ObjectiveType
from Utils.RelationalOperators import RelationalOperators
from Utils.Stochasticity import Stochasticity
from Validator.Validator import Validator
from ValueGenerator.ValueGenerator import ValueGenerator


class Naive:

    def __init__(self, query: Query,
                 linear_relaxation: bool,
                 dbInfo: DbInfo,
                 init_no_of_scenarios: int,
                 no_of_validation_scenarios: int,
                 approximation_bound: float):
        self.__query = query
        self.__gurobi_env = gp.Env(
            params=GurobiLicense.OPTIONS)
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
            no_of_validation_scenarios
        )
        self.__approximation_bound = \
            approximation_bound

        self.__no_of_vars = \
            self.__get_number_of_tuples()
        self.__feasible_no_of_scenarios_to_store = \
            int(np.floor(
                (0.50*psutil.virtual_memory().available)/\
                (64*self.__no_of_vars)))
        
        self.__vars = []

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

    def __add_var_constraint_to_model(
        self, var_constraint,
        no_of_scenarios
    ):
        inequality_sign = \
            var_constraint.get_inequality_sign()
        
        gurobi_inequality = GRB.LESS_EQUAL
        if inequality_sign == RelationalOperators.EQUALS:
            gurobi_inequality = GRB.EQUAL
        elif inequality_sign == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            gurobi_inequality = GRB.GREATER_EQUAL
        
        sum_limit = var_constraint.get_sum_limit()
        
        indicators = []

        for _ in range(no_of_scenarios):
            if (_ % \
                self.__feasible_no_of_scenarios_to_store
            ) == 0:
                self.__add_feasible_no_of_scenarios(
                    var_constraint.get_attribute_name()
                )
            
            scenario_index = _ % \
                self.__feasible_no_of_scenarios_to_store

            coefficients = []
            for ind in range(self.__no_of_vars):
                coefficients.append(
                    self.__scenarios[
                        var_constraint.get_attribute_name()
                    ][ind][scenario_index]
                )
            indicator = self.__model.addVar(vtype=GRB.BINARY)
            self.__model.addGenConstrIndicator(
                indicator, True,
                gp.LinExpr(coefficients, self.__vars),
                gurobi_inequality, sum_limit
            )
            indicators.append(indicator)
        
        self.__model.addConstr(
            gp.LinExpr([1]*no_of_scenarios, indicators),
            GRB.GREATER_EQUAL,
            no_of_scenarios * \
                var_constraint.get_probability_threshold()
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


    def __add_constraints_to_model(
        self, no_of_scenarios, probabilistically_constrained):
        if probabilistically_constrained:
            self.__add_scenarios_if_necessary(no_of_scenarios)

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
            if constraint.is_var_constraint():
                if probabilistically_constrained:
                    self.__add_var_constraint_to_model(
                        constraint, no_of_scenarios
                    ) 

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
            probabilistically_constrained = True):
        
        self.__model = gp.Model(
            env=self.__gurobi_env)
        self.__add_variables_to_model()
        self.__add_constraints_to_model(
            no_of_scenarios, probabilistically_constrained)
        self.__add_objective_to_model(
            self.__query.get_objective(), no_of_scenarios)
    
    
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


    def solve(self):
        no_of_scenarios = self.__init_no_of_scenarios
        
        self.__model_setup(
            no_of_scenarios,
            probabilistically_constrained=False)
        
        probabilistically_unconstrained_package = \
            self.__get_package()
        
        if probabilistically_unconstrained_package is None:
            return None

        upper_bound = \
            self.__validator.get_validation_objective_value(
                package_dict=probabilistically_unconstrained_package
            )
        
        if self.__validator.is_package_validation_feasible(
            probabilistically_unconstrained_package):
            print('Probabilistically unconstrained package is feasible')
            for constraint in self.__query.get_constraints():
                if constraint.is_var_constraint():
                    print('Passes', self.__validator.get_var_constraint_satisfaction(
                        package_dict=probabilistically_unconstrained_package,
                        var_constraint=constraint
                    ), 'of scenarios')
            return (probabilistically_unconstrained_package,
                    upper_bound)
        
        while no_of_scenarios <= self.__no_of_validation_scenarios:
            print('Trying with', no_of_scenarios, 'scenarios')
            self.__model_setup(no_of_scenarios)
            package = self.__get_package()

            if package is None:
                ...
            
            elif self.__validator.is_package_validation_feasible(package):
                if self.__validator.is_package_1_pm_epsilon_approximate(
                    package, self.__approximation_bound, upper_bound):
                    return (
                        package, 
                        self.__validator.get_validation_objective_value(
                            package))
            
            else:
                objective_value = \
                    self.__validator.get_validation_objective_value(
                        package
                    )
                if self.__query.get_objective().get_objective_type() \
                    == ObjectiveType.MAXIMIZATION:
                    if objective_value < upper_bound:
                        upper_bound = objective_value
                elif objective_value > upper_bound:
                    upper_bound = objective_value

            no_of_scenarios *= 2
        
        return None


