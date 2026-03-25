import math
import numpy as np
import time


from DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGenerator import RepresentativeScenarioGenerator
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGeneratorWithoutCorrelation import RepresentativeScenarioGeneratorWithoutCorrelation
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraint import VaRConstraint
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from StochasticPackageQuery.Query import Query
from Utils.RelationalOperators import RelationalOperators
from Utils.Relation_Prefixes import Relation_Prefixes
from Utils.TailType import TailType
from Utils.ObjectiveType import ObjectiveType


class RefineValidator:

    def __init__(
        self, query: Query,
        dbInfo: DbInfo,
        no_of_validation_scenarios: int,
        partition_validation_scenarios,
        chosen_tuple_validation_scenarios,
        partition_variable_multiplicities,
        chosen_tuple_multiplicities
    ):
        self.__query = query
        self.__dbInfo = dbInfo
        self.__no_of_validation_scenarios =\
            no_of_validation_scenarios
        self.__partition_validation_scenarios = partition_validation_scenarios
        self.__chosen_tuple_validation_scenarios = chosen_tuple_validation_scenarios
        self.__partition_variable_multiplicities = partition_variable_multiplicities
        self.__chosen_tuple_multiplicities = chosen_tuple_multiplicities
        self.__scenario_cache = {}

    def __get_scenarios_and_ids(
        self, package_dict, attribute
    ):
        cache_key = (attribute, tuple(sorted(package_dict.items())))
        if cache_key in self.__scenario_cache:
            return self.__scenario_cache[cache_key]
        base_predicate = ''
        ids_with_multiplicities = []
        for id in package_dict:
            ids_with_multiplicities.append((id, package_dict[id]))
            if len(base_predicate) > 0:
                base_predicate += " or "
            base_predicate += " id = " + str(id)
        ids_with_multiplicities.sort()
        if len(package_dict) > 0:
            scenarios = \
                self.__dbInfo.get_variable_generator_function(attribute)(
                    relation=self.__query.get_relation(),
                    base_predicate=base_predicate
                ).generate_scenarios(
                    seed=Hyperparameters.VALIDATION_SEED,
                    no_of_scenarios=self.__no_of_validation_scenarios,
                )
        else:
            scenarios = []
        result = (scenarios, ids_with_multiplicities)
        self.__scenario_cache[cache_key] = result
        return result

    def __compute_scenario_scores(self, attribute, scenarios, ids_with_multiplicities):
        rows = list(scenarios)
        mults_list = [m for _, m in ids_with_multiplicities]
        for partition in self.__partition_validation_scenarios[attribute]:
            for dup_idx, s in enumerate(
                    self.__partition_validation_scenarios[attribute][partition]):
                rows.append(s)
                mults_list.append(
                    self.__partition_variable_multiplicities[partition][dup_idx])
        for t in self.__chosen_tuple_validation_scenarios[attribute]:
            for s in self.__chosen_tuple_validation_scenarios[attribute][t]:
                rows.append(s)
                mults_list.append(self.__chosen_tuple_multiplicities[t])
        return np.array(rows).T @ np.array(mults_list)

    def get_validation_objective_value(self, package_dict) -> float:
        if package_dict is None:
            if self.__query.get_objective().get_objective_type() == \
                ObjectiveType.MAXIMIZATION:
                return -math.inf
            else:
                return math.inf
        
        if len(package_dict) == 0:
            return 0

        attribute = self.__query.get_objective().get_attribute_name()
        scenarios, ids_with_multiplicities = self.__get_scenarios_and_ids(
            package_dict, attribute)
        idx = 0
        objective_value = 0
        for tuple_values in scenarios:
            _, multiplicity = ids_with_multiplicities[idx]
            idx += 1
            objective_value += np.average(tuple_values)*multiplicity
        
        for partition in self.__partition_validation_scenarios[attribute]:
            for dup_idx, tuple_scenario in enumerate(
                self.__partition_validation_scenarios[attribute][partition]
            ):
                objective_value += np.average(tuple_scenario)*\
                    self.__partition_variable_multiplicities[partition][dup_idx]

        for tuple in self.__chosen_tuple_validation_scenarios[attribute]:
            for tuple_scenarios in self.__chosen_tuple_validation_scenarios[attribute][tuple]:
                objective_value += np.average(tuple_scenarios)*\
                    self.__chosen_tuple_multiplicities[tuple]
        return objective_value

    def get_expected_sum_constraint_feasibility(
        self, package_dict, expected_sum_constraint: ExpectedSumConstraint
    ) -> bool:
        if package_dict is None:
            return True

        attribute = expected_sum_constraint.get_attribute_name()
        scenarios, ids_with_multiplicities =\
            self.__get_scenarios_and_ids(package_dict, attribute)
        idx = 0
        expected_sum = 0
        for scenario in scenarios:
            _, multiplicity = ids_with_multiplicities[idx]
            idx += 1
            expected_sum += np.average(scenario)*multiplicity
        
        for partition in self.__partition_validation_scenarios[attribute]:
            for dup_idx, tuple_scenarios in enumerate(
                self.__partition_validation_scenarios[attribute][partition]
            ):
                expected_sum += np.average(tuple_scenarios)*\
                    self.__partition_variable_multiplicities[partition][dup_idx]

        for tuple in self.__chosen_tuple_validation_scenarios[attribute]:
            for tuple_scenarios in self.__chosen_tuple_validation_scenarios[attribute][tuple]:
                expected_sum += np.average(tuple_scenarios)*\
                    self.__chosen_tuple_multiplicities[tuple]
        
        if expected_sum_constraint.get_inequality_sign() ==\
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            return expected_sum >=\
                expected_sum_constraint.get_sum_limit()
        
        if expected_sum_constraint.get_inequality_sign() ==\
            RelationalOperators.LESS_THAN_OR_EQUAL_TO:
            return expected_sum <=\
                expected_sum_constraint.get_sum_limit()

        return expected_sum == expected_sum_constraint.get_sum_limit()


    def get_var_among_validation_scenarios(
        self, package_dict, var_constraint: VaRConstraint
    ) -> float:
        attribute = var_constraint.get_attribute_name()
        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute
            )
        
        scenario_scores = np.sort(
            self.__compute_scenario_scores(attribute, scenarios, ids_with_multiplicities)
        )[::-1]
        scenarios_to_consider = \
            min(
                int(np.floor((var_constraint.get_probability_threshold()*\
                              self.__no_of_validation_scenarios))),
                self.__no_of_validation_scenarios - 1
            )
        return float(scenario_scores[scenarios_to_consider])
    
    def get_var_constraint_satisfaction(
        self, package_dict, var_constraint: VaRConstraint
    ) -> float:
        if package_dict is None:
            return 1.0
        attribute = var_constraint.get_attribute_name()
        scenarios, ids_with_multiplicities =\
            self.__get_scenarios_and_ids(
                package_dict, attribute
            )
        scenario_scores = self.__compute_scenario_scores(
            attribute, scenarios, ids_with_multiplicities)
        limit = var_constraint.get_sum_limit()
        if var_constraint.get_inequality_sign() ==\
                RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            satisfying_scenarios = int(np.sum(scenario_scores >= limit))
        elif var_constraint.get_inequality_sign() ==\
                RelationalOperators.LESS_THAN_OR_EQUAL_TO:
            satisfying_scenarios = int(np.sum(scenario_scores <= limit))
        else:
            satisfying_scenarios = int(np.sum(scenario_scores == limit))
        return satisfying_scenarios / self.__no_of_validation_scenarios
    
    def get_cvar_among_validation_scenarios(
        self, package_dict,
        cvar_constraint: CVaRConstraint
    ):
        if package_dict is None:
            if cvar_constraint.get_inequality_sign() ==\
                RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                return math.inf
            else:
                return -math.inf

        attribute = cvar_constraint.get_attribute_name()

        scenarios, ids_with_multiplicities =\
            self.__get_scenarios_and_ids(
                package_dict, attribute
            )
        scenario_scores = self.__compute_scenario_scores(
            attribute, scenarios, ids_with_multiplicities)
        if cvar_constraint.get_tail_type() == TailType.HIGHEST:
            scenario_scores = np.sort(scenario_scores)[::-1]
        else:
            scenario_scores = np.sort(scenario_scores)

        no_of_scenarios_to_consider = int(np.floor(
            self.__no_of_validation_scenarios*\
                cvar_constraint.get_percentage_of_scenarios()/100.0
        ))

        return float(np.average(scenario_scores[0: no_of_scenarios_to_consider]))

    def get_var_constraint_feasibility(
        self, package_dict,
        var_constraint: VaRConstraint
    ) -> bool:
        if package_dict is None:
            return True
        probability = self.get_var_constraint_satisfaction(
            package_dict, var_constraint
        )
        return probability >= var_constraint.get_probability_threshold()
    
    def get_cvar_constraint_feasibility(
        self, package_dict, cvar_constraint: CVaRConstraint
    ) -> bool:
        if package_dict is None:
            return True
        cvar = self.get_cvar_among_validation_scenarios(
            package_dict, cvar_constraint
        )

        if cvar_constraint.get_inequality_sign() ==\
            RelationalOperators.LESS_THAN_OR_EQUAL_TO:
            return cvar <= cvar_constraint.get_sum_limit()
        
        elif cvar_constraint.get_inequality_sign() ==\
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            return cvar >= cvar_constraint.get_sum_limit()
    
        return cvar == cvar_constraint.get_sum_limit()
    
    def is_package_validation_feasible(
        self, package_dict
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
        
        objective = self.__query.get_objective()

        objective_value = self.get_validation_objective_value(
            package_dict
        )
        
        if objective.get_objective_type() == ObjectiveType.MAXIMIZATION:
            return objective_value >= (1-epsilon)*upper_bound
        
        return objective_value <= (1 + epsilon)*upper_bound