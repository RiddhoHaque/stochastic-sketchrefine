from DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraint import VaRConstraint
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from StochasticPackageQuery.Query import Query
from Utils.RelationalOperators import RelationalOperators
from Utils.TailType import TailType
from Utils.ObjectiveType import ObjectiveType
import math
import numpy as np


class Validator:

    def __init__(self,query: Query,
                 dbInfo: DbInfo,
                 no_of_validation_scenarios: int):
        self.__query = query
        self.__dbInfo = dbInfo
        self.__no_of_validation_scenarios = \
            no_of_validation_scenarios
    

    def __get_scenarios_and_ids(self, package_dict: dict,
                              attribute: str):
        base_predicate = ''
        ids_with_multiplicities = []
        for id in package_dict:
            ids_with_multiplicities.append(
                (id, package_dict[id]))
            if len(base_predicate) > 0:
                base_predicate += " or "
            base_predicate += " id=" + str(id)
        ids_with_multiplicities.sort()
        if len(package_dict) > 0:
            scenarios = \
                self.__dbInfo.get_variable_generator_function(
                    attribute)(
                        relation=self.__query.get_relation(),
                        base_predicate=base_predicate
                    ).generate_scenarios(
                        seed=Hyperparameters.VALIDATION_SEED,
                        no_of_scenarios = self.__no_of_validation_scenarios,
                    )
        else:
            scenarios = []
        return scenarios, ids_with_multiplicities


    def get_validation_objective_value(self, package_dict) -> float:
        if package_dict is None:
            if self.__query.get_objective().get_objective_type() == \
                ObjectiveType.MAXIMIZATION:
                return -math.inf
            else:
                return math.inf
        
        if len(package_dict) == 0:
            return 0
            
        attribute = \
            self.__query.get_objective().get_attribute_name()
        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute)
        idx = 0
        objective_value = 0
        for tuple_values in scenarios:
            _, multiplicity = ids_with_multiplicities[idx]
            idx += 1
            #print('Validation Average:', np.average(tuple_values))
            #print('Validation multiplicity:', multiplicity)
            objective_value += np.average(tuple_values)*multiplicity
        
        return objective_value
    

    def get_expected_sum_constraint_feasibility(
            self, package_dict,
            expected_sum_constraint: ExpectedSumConstraint) -> bool:
        if package_dict is None:
            return True
        
        attribute = expected_sum_constraint.get_attribute_name()
        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute)
        idx = 0
        expected_sum = 0
        for scenario in scenarios:
            _, multiplicity = ids_with_multiplicities[idx]
            idx += 1
            expected_sum += np.sum(scenario)*multiplicity
        expected_sum /= self.__no_of_validation_scenarios

        if expected_sum_constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            return expected_sum >= \
                expected_sum_constraint.get_sum_limit()
        
        if expected_sum_constraint.get_inequality_sign() == \
            RelationalOperators.EQUALS:
            return expected_sum == \
                expected_sum_constraint.get_sum_limit()
        
        return expected_sum <= \
            expected_sum_constraint.get_sum_limit()
    

    def get_var_among_validation_scenarios(
        self, package_dict,
        var_constraint: VaRConstraint
    ) -> float:
        attribute = var_constraint.get_attribute_name()
        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute
            )
        
        scenario_scores = []
        for _ in range(self.__no_of_validation_scenarios):
            idx = 0
            scenario_score = 0
            for __, multiplicity in ids_with_multiplicities:
                scenario_score += \
                    scenarios[idx][_] * multiplicity
                idx += 1
            scenario_scores.append(
                scenario_score
            )
        scenario_scores.sort()
        scenarios_to_consider = \
            int(np.floor((var_constraint.get_probability_threshold()*\
                      self.__no_of_validation_scenarios)))
        return scenario_scores[scenarios_to_consider]


    def get_var_constraint_satisfaction(
            self, package_dict,
            var_constraint: VaRConstraint
    ) -> float:
        if package_dict is None:
            return 1.00
        attribute = var_constraint.get_attribute_name()
        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute
            )
        satisfying_scenarios = 0
        for _ in range(self.__no_of_validation_scenarios):
            idx = 0
            scenario_score = 0
            for __, multiplicity in ids_with_multiplicities:
                scenario_score += \
                    scenarios[idx][_] * multiplicity
                idx += 1
            if var_constraint.get_inequality_sign() == \
                RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                if scenario_score >= var_constraint.get_sum_limit():
                    satisfying_scenarios += 1
            elif var_constraint.get_inequality_sign() == \
                RelationalOperators.LESS_THAN_OR_EQUAL_TO:
                if scenario_score <= var_constraint.get_sum_limit():
                    satisfying_scenarios += 1
        return satisfying_scenarios / self.__no_of_validation_scenarios
    

    def get_cvar_constraint_satisfaction(
        self, package_dict,
        cvar_constraint: CVaRConstraint 
    ) -> float:
        if package_dict is None:
            if cvar_constraint.get_inequality_sign() == \
                RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                return math.inf
            else:
                return -math.inf
        attribute = cvar_constraint.get_attribute_name()
        
        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute
            )
        scenario_scores = []
        for scenario_no in range(self.__no_of_validation_scenarios):
            idx = 0
            scenario_score = 0
            for __, multiplicity in ids_with_multiplicities:
                scenario_score += \
                    scenarios[idx][scenario_no] * multiplicity
                idx += 1
            scenario_scores.append(scenario_score)
        
        if cvar_constraint.get_tail_type() == TailType.HIGHEST:
            scenario_scores.sort(reverse=True)
        else:
            scenario_scores.sort()

        no_of_scenarios_to_consider = \
            int(
                np.floor(
                    self.__no_of_validation_scenarios*\
                    cvar_constraint.get_percentage_of_scenarios()\
                        /100
                )
            )

        return np.average(scenario_scores[
            0: no_of_scenarios_to_consider])    


    def get_var_constraint_feasibility(
        self, package_dict,
        var_constraint: VaRConstraint
    ) -> bool:
        if package_dict is None:
            return True
        probability = \
            self.get_var_constraint_satisfaction(
                package_dict, var_constraint
            )
        return (
            probability >= \
            var_constraint.get_probability_threshold()
        )

    def get_cvar_constraint_feasibility(
        self, package_dict,
        cvar_constraint: CVaRConstraint
    ) -> bool:
        if package_dict is None:
            return True
        cvar = \
            self.get_cvar_constraint_satisfaction(
                package_dict, cvar_constraint,
            )
        
        if cvar_constraint.get_inequality_sign() == \
            RelationalOperators.LESS_THAN_OR_EQUAL_TO:
            return (cvar <= cvar_constraint.get_sum_limit())
        
        elif cvar_constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            return (cvar >= cvar_constraint.get_sum_limit())
        
        return cvar == cvar_constraint.get_sum_limit()
    

    def is_package_validation_feasible(
        self, package_dict,
    ):
        if package_dict is None:
            return True
        
        for constraint in self.__query.get_constraints():
            if constraint.is_expected_sum_constraint():
                if not self.get_expected_sum_constraint_feasibility(
                    package_dict, constraint
                ):
                    return False
            if constraint.is_var_constraint():
                if not self.get_var_constraint_feasibility(
                    package_dict, constraint
                ):
                    return False
                
            if constraint.is_cvar_constraint():
                if not self.get_cvar_constraint_feasibility(
                    package_dict, constraint
                ):
                    return False
        return True


    def is_package_1_pm_epsilon_approximate(
        self, package_dict: dict,
        epsilon: float, upper_bound: float
    ):
        if package_dict is None:
            return False

        objective = \
            self.__query.get_objective()
        
        objective_value = \
            self.get_validation_objective_value(
                package_dict
            )
        if objective.get_objective_type() == \
            ObjectiveType.MAXIMIZATION:
            return objective_value >= \
                (1 - epsilon) * upper_bound
        
        return objective_value <= \
            (1 + epsilon) * upper_bound

    