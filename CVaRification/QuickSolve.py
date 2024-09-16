import gurobipy as gp
from gurobipy import GRB
import numpy as np
import psutil

from CVaRification.CVaRificationSearchResults import CVaRificationSearchResults
from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from SeedManager.SeedManager import SeedManager
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraint import VaRConstraint
from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraint import DeterministicConstraint
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


class QuickSolve:
    
    def __init__(
        self, query: Query,
        dbInfo: DbInfo,
        init_no_of_scenarios: int,
        no_of_validation_scenarios: int,
        sampling_tolerance: float,
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
            False
        self.__init_no_of_scenarios = \
            init_no_of_scenarios
        
        self.__no_of_validation_scenarios = \
            no_of_validation_scenarios 
        self.__validator = Validator(
            self.__query, dbInfo,
            self.__no_of_validation_scenarios
        )
        self.__sampling_tolerance = \
            sampling_tolerance
        self.__no_of_vars = \
            self.__get_number_of_tuples()
        self.__feasible_no_of_scenarios_to_store = \
            int(np.floor(
                (0.40*psutil.virtual_memory().available)/\
                (64*self.__no_of_vars)
            ))

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

    
    def __get_stochastic_attributes(self):
        attributes = set()
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
    
    
    def __add_constraints_to_model(
        self, no_of_scenarios: int,
    ):
        self.__add_all_scenarios_if_possible(
            no_of_scenarios
        )
        for constraint in self.__query.get_constraints():
            if constraint.is_package_size_constraint():
                self.__add_package_size_constraint_to_model(
                    constraint
                )
            if constraint.is_deterministic_constraint():
                self.__add_deterministic_constraint_to_model(
                    constraint
                )
    

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
    ):
        
        self.__model = gp.Model(
            env=self.__gurobi_env)
        self.__add_variables_to_model()
        self.__add_constraints_to_model(
            no_of_scenarios)
        self.__add_objective_to_model(
            self.__query.get_objective(),
            no_of_scenarios)
    
    
    def __get_package(self):
        self.__model.optimize()
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


    def solve(self, can_add_scenarios = True):
        no_of_scenarios = self.__init_no_of_scenarios
        unacceptable_diff = True

        while unacceptable_diff:
            self.__model_setup(
                no_of_scenarios=no_of_scenarios,
            )

            probabilistically_unconstrained_package = \
                self.__get_package()
            print('Probabilistically unconstrained package:',
                probabilistically_unconstrained_package)
            if probabilistically_unconstrained_package is None:
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
            return (probabilistically_unconstrained_package,
                    objective_upper_bound, True)
        
        return (probabilistically_unconstrained_package,
                objective_upper_bound, False)