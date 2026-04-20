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
from Utils.Stochasticity import Stochasticity


class SketchValidator:

    def __init__(
        self, query: Query,
        dbInfo: DbInfo,
        no_of_validation_scenarios: int,
        maxed_out_duplicate_vector: list[int],
        partition_id_in_duplicate_vector: dict,
        partition_ids: list[int] = None
    ):
        self.__query = query
        self.__dbInfo = dbInfo
        self.__no_of_validation_scenarios = \
            no_of_validation_scenarios
        self.__maxed_out_duplicate_vector = \
            list(maxed_out_duplicate_vector)
        self.__partition_ids = partition_ids
        self.__partition_id_in_duplicate_vector = \
            partition_id_in_duplicate_vector
        self.__scenario_cache = {}
        self.__representative_value_cache = {}

        # Map absolute partition_id -> filtered index
        if partition_ids is not None:
            self.__pid_to_idx = {pid: idx for idx, pid in enumerate(partition_ids)}
            partition_filter = ' AND partition_id IN (' + \
                ','.join(str(p) for p in partition_ids) + ')'
        else:
            self.__pid_to_idx = None
            partition_filter = ''

        self.__prefix_sum_max_duplicates = [0]

        for max_duplicates in self.__maxed_out_duplicate_vector:
            self.__prefix_sum_max_duplicates.append(
                self.__prefix_sum_max_duplicates[-1] + \
                    max_duplicates
            )

        if self.__dbInfo.has_inter_tuple_correlations():
            attributes = self.__dbInfo.get_stochastic_attributes()

            self.__correlations = dict()

            for attribute in attributes:
                relation = self.__query.get_relation()

                # Get all correlations and arrange them via pids
                # according to their maxed out duplicates

                correlation_relation = \
                    Relation_Prefixes.INIT_CORRELATION_PREFIX +\
                    relation

                sql = "SELECT partition_id, duplicates, init_corr " +\
                    " FROM " + correlation_relation +\
                    " WHERE attribute='" + attribute + "'" +\
                    partition_filter +\
                    " ORDER BY (partition_id, duplicates);"

                PgConnection.Execute(sql)
                tuples = PgConnection.Fetch()

                self.__correlations[attribute] = \
                    [[] for _ in range(len(
                        self.__maxed_out_duplicate_vector))]

                for tuple in tuples:
                    abs_pid, duplicates, init_corr = tuple
                    idx = self.__pid_to_idx[abs_pid] \
                        if self.__pid_to_idx is not None else abs_pid
                    assert duplicates == len(self.__correlations[attribute][idx]) + 1
                    self.__correlations[attribute][idx].append(init_corr)

                for idx in range(len(self.__correlations[attribute])):
                    actual = len(self.__correlations[attribute][idx])
                    if actual < self.__maxed_out_duplicate_vector[idx]:
                        self.__maxed_out_duplicate_vector[idx] = actual
                    elif actual > self.__maxed_out_duplicate_vector[idx]:
                        self.__correlations[attribute][idx] = \
                            self.__correlations[attribute][idx][:self.__maxed_out_duplicate_vector[idx]]
                print('Got initial correlations')

            # Recompute prefix sums in case clamping changed any entry.
            self.__prefix_sum_max_duplicates = [0]
            for max_duplicates in self.__maxed_out_duplicate_vector:
                self.__prefix_sum_max_duplicates.append(
                    self.__prefix_sum_max_duplicates[-1] + max_duplicates
                )

        attributes = self.__dbInfo.get_stochastic_attributes()
        self.__representatives = dict()
        relation = Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX +\
            self.__query.get_relation()

        for attribute in attributes:
            sql = 'SELECT partition_id, representative_tuple_id ' +\
                'FROM ' + relation + " WHERE attribute='" + attribute +\
                "'" + partition_filter + " ORDER BY partition_id;"
            PgConnection.Execute(sql)
            tuples = PgConnection.Fetch()

            n = len(self.__maxed_out_duplicate_vector)
            self.__representatives[attribute] = [None] * n

            for tuple in tuples:
                abs_pid, representative_id = tuple
                idx = self.__pid_to_idx[abs_pid] \
                    if self.__pid_to_idx is not None else abs_pid
                self.__representatives[attribute][idx] = representative_id
        #print('Initialized sketch validator')

    def __to_integral_count(self, multiplicity: float) -> int:
        rounded = round(multiplicity)
        if abs(multiplicity - rounded) <= 1e-6:
            return int(rounded)
        return int(multiplicity)

    def __get_materialized_duplicate_count(
        self,
        partition_idx: int,
        multiplicity: float
    ) -> int:
        max_duplicates = self.__maxed_out_duplicate_vector[partition_idx]
        if max_duplicates <= 0 or multiplicity <= 1e-9:
            return 0

        integral = self.__to_integral_count(multiplicity)
        if abs(multiplicity - integral) <= 1e-6:
            requested = integral
        else:
            requested = int(np.ceil(multiplicity - 1e-9))

        return min(max_duplicates, max(1, requested))

    def __split_multiplicity_across_duplicates(
        self,
        multiplicity: float,
        duplicates: int
    ) -> list[float | int]:
        if duplicates <= 0:
            return []

        integral = self.__to_integral_count(multiplicity)
        if abs(multiplicity - integral) <= 1e-6:
            distribution = [integral // duplicates for _ in range(duplicates)]
            for idx in range(integral % duplicates):
                distribution[idx] += 1
            return distribution

        base = np.floor(multiplicity / duplicates)
        distribution = [float(base) for _ in range(duplicates)]
        remaining = multiplicity - (base * duplicates)
        idx = 0
        while remaining > 1e-9 and idx < duplicates:
            increment = min(1.0, remaining)
            distribution[idx] += increment
            remaining -= increment
            idx += 1
        if remaining > 1e-9:
            distribution[-1] += remaining
        return distribution

    def update_partition_id_in_duplicate_vector(
        self, partition_id_in_duplicate_vector: dict
    ):
        #print('Updating Partition ID in duplicate vector')
        self.__partition_id_in_duplicate_vector =\
            partition_id_in_duplicate_vector
        self.__scenario_cache = {}

    def __get_partition_index(self, partition_id: int) -> int:
        if self.__pid_to_idx is None:
            return partition_id
        return self.__pid_to_idx[partition_id]

    def bump_correlations_for_partitions(
        self, local_pids: list, delta: float = 0.1
    ) -> bool:
        if not self.__dbInfo.has_inter_tuple_correlations():
            return False
        any_increased = False
        for attribute in self.__correlations:
            for pid in local_pids:
                idx = self.__get_partition_index(pid)
                for d in range(len(self.__correlations[attribute][idx])):
                    old = self.__correlations[attribute][idx][d]
                    new_val = min(1.0, old + delta)
                    if new_val > old:
                        any_increased = True
                    self.__correlations[attribute][idx][d] = new_val
        self.__scenario_cache.clear()
        return any_increased

    def __get_partition_multiplicities(
        self, package_dict: dict
    ) -> list[tuple[int, float]]:
        pid_to_multiplicity = {}
        for id in package_dict:
            pid = self.__partition_id_in_duplicate_vector[id]
            if pid not in pid_to_multiplicity:
                pid_to_multiplicity[pid] = 0.0
            pid_to_multiplicity[pid] += package_dict[id]
        return sorted(pid_to_multiplicity.items())

    def __get_representative_values(
        self, attribute: str
    ) -> list[float]:
        if attribute in self.__representative_value_cache:
            return self.__representative_value_cache[attribute]

        relation = self.__query.get_relation()
        representative_relation = \
            Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX + relation
        sql = 'SELECT r.partition_id, t.' + attribute + \
            ' FROM ' + representative_relation + ' AS r INNER JOIN ' + \
            relation + ' AS t ON r.representative_tuple_id=t.id WHERE ' + \
            "r.attribute='" + attribute + "'"
        if self.__partition_ids is not None:
            sql += ' AND r.partition_id IN (' + \
                ','.join(str(p) for p in self.__partition_ids) + ')'
        sql += ' ORDER BY r.partition_id;'

        PgConnection.Execute(sql)
        tuples = PgConnection.Fetch()

        values = [0.0] * len(self.__maxed_out_duplicate_vector)
        for abs_pid, representative_value in tuples:
            idx = self.__get_partition_index(abs_pid)
            values[idx] = representative_value

        self.__representative_value_cache[attribute] = values
        return values

    def __get_scenarios_and_ids(self, package_dict: dict,
                              attribute: str,
                              consider_correlation: bool):
        cache_key = (attribute, consider_correlation,
                     tuple(sorted(package_dict.items())))
        if cache_key in self.__scenario_cache:
            return self.__scenario_cache[cache_key]
        base_predicate = ''
        pids_with_multiplicities = \
            self.__get_partition_multiplicities(package_dict)
        for pid, _ in pids_with_multiplicities:
            if len(base_predicate) > 0:
                base_predicate += " or "
            base_predicate += " partition_id=" + str(pid)
        #print('PIDs with multiplicities:',
        #      pids_with_multiplicities)

        if consider_correlation and \
            self.__dbInfo.has_inter_tuple_correlations():
            #print('Generating Validation Scenarios'
            #      'with correlations')
            duplicates = []
            correlations = []
            pids = []
            representatives = []
            for pid, multiplicity in pids_with_multiplicities:
                idx = self.__get_partition_index(pid)
                duplicate_count = self.__get_materialized_duplicate_count(
                    idx, multiplicity
                )
                if duplicate_count == 0:
                    continue
                duplicates.append(duplicate_count)
                #print('Number of Duplicates:', duplicates[-1])
                #print('Correlations available for',
                #      len(self.__correlations[attribute][idx]),
                #      'duplicates')
                correlations.append(
                    self.__correlations[attribute][idx][duplicates[-1]-1])
                #print('Appended correlation:', correlations[-1])
                pids.append(pid)
                representatives.append(
                    self.__representatives[attribute][idx]
                )
                #print('Representative:', self.__representatives[attribute][idx])
        else:
            #print('Generating Scenarios without correlation')
            duplicates = []
            representatives = []
            for pid, multiplicity in pids_with_multiplicities:
                idx = self.__get_partition_index(pid)
                duplicate_count = self.__get_materialized_duplicate_count(
                    idx, multiplicity
                )
                if duplicate_count == 0:
                    continue
                duplicates.append(duplicate_count)
                representatives.append(
                    self.__representatives[attribute][idx]
                )
                #print('PID:', pid)
                #print('Duplicates:', duplicates[-1])
                #print('Representatives:', representatives[-1])

        if len(package_dict) > 0:
            if consider_correlation and \
                self.__dbInfo.has_inter_tuple_correlations():
                #print('Generating Scenarios with correlation')
                start_time = time.time()
                scenario_generator = \
                    RepresentativeScenarioGenerator(
                        relation=self.__query.get_relation(),
                        dbInfo=self.__dbInfo,
                        attr=attribute,
                    )
                scenarios = \
                    scenario_generator.generate_scenarios_multiple_pids(
                        seed=Hyperparameters.VALIDATION_SEED,
                        no_of_scenarios=self.__no_of_validation_scenarios,
                        pids = pids, duplicates=duplicates,
                        correlations_list=correlations,
                    )
                #print('Validation scenarios generated in',
                #      time.time() - start_time, 'seconds')
            else:
                start_time = time.time()
                #print('Generating Scenarios without correlation')
                scenarios = \
                    RepresentativeScenarioGeneratorWithoutCorrelation(
                        relation=self.__query.get_relation(),
                        attr=attribute,
                        base_predicate=base_predicate,
                        duplicates=duplicates,
                        representatives=representatives,
                        scenario_generator=self.\
                            __dbInfo.get_variable_generator_function(
                                attribute)
                    ).generate_scenarios(
                        seed=Hyperparameters.VALIDATION_SEED,
                        no_of_scenarios=self.__no_of_validation_scenarios
                    )
                assert len(scenarios) == np.sum(duplicates)
                for tuple_scenarios in scenarios:
                    assert len(tuple_scenarios) == \
                        self.__no_of_validation_scenarios
                #print('Validation scenarios generated in',
                #      time.time() - start_time, 'seconds')
        else:
            scenarios = []
        
        ids_with_multiplicities = []

        for pid, multiplicity in pids_with_multiplicities:
            idx = self.__get_partition_index(pid)
            duplicates = self.__get_materialized_duplicate_count(
                idx, multiplicity
            )
            if duplicates == 0:
                continue
            init_index = self.__prefix_sum_max_duplicates[idx]

            #print('No. of duplicates:', duplicates)
            for duplicate, duplicate_multiplicity in enumerate(
                self.__split_multiplicity_across_duplicates(
                    multiplicity, duplicates
                )
            ):
                id = init_index + duplicate
                #print('Duplicate', duplicate,
                #      'multiplicity:', multiplicity)
                ids_with_multiplicities.append(
                    (id, duplicate_multiplicity))

        assert len(ids_with_multiplicities) == len(scenarios)
        result = (scenarios, ids_with_multiplicities)
        self.__scenario_cache[cache_key] = result
        return result


    def get_validation_objective_value(self, package_dict) -> float:
        if package_dict is None:
            if self.__query.get_objective().get_objective_type() == \
                ObjectiveType.MAXIMIZATION:
                #print('Null package, returning obj. value -inf')
                return -math.inf
            else:
                #print('Null package, returning obj. value inf')
                return math.inf

        if len(package_dict) == 0:
            #print('Empty package, returning obj. value 0')
            return 0

        objective = self.__query.get_objective()
        attribute = objective.get_attribute_name()

        if objective.is_cvar_objective():
            scenarios, ids_with_multiplicities = \
                self.__get_scenarios_and_ids(
                    package_dict, attribute, True)
            mat = np.array(scenarios)
            mults = np.array([m for _, m in ids_with_multiplicities])
            scenario_scores = mat.T @ mults
            tail_type = objective.get_tail_type()
            if tail_type == TailType.HIGHEST:
                scenario_scores = np.sort(scenario_scores)[::-1]
            else:
                scenario_scores = np.sort(scenario_scores)
            k = max(1, int(np.floor(
                self.__no_of_validation_scenarios *
                objective.get_percentage_of_scenarios()
            )))
            return float(np.average(scenario_scores[:k]))

        if objective.get_stochasticity() == Stochasticity.DETERMINISTIC:
            representative_values = \
                self.__get_representative_values(attribute)
            objective_value = 0.0
            for pid, multiplicity in self.__get_partition_multiplicities(
                package_dict
            ):
                idx = self.__get_partition_index(pid)
                objective_value += representative_values[idx] * multiplicity
            return objective_value

        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute, False)
        idx = 0
        objective_value = 0
        for tuple_values in scenarios:
            _, multiplicity = ids_with_multiplicities[idx]
            idx += 1
            #print('Validation Average:', np.average(tuple_values))
            #print('Validation multiplicity:', multiplicity)
            objective_value += np.average(tuple_values)*multiplicity
        #print('Obj. Value:', objective_value)
        return objective_value
    

    def get_expected_sum_constraint_feasibility(
        self, package_dict,
        expected_sum_constraint: ExpectedSumConstraint
    ) -> bool:
        if package_dict is None:
            return True
        
        attribute = expected_sum_constraint.get_attribute_name()
        #print('Getting scenarios and ids')
        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute,
                    consider_correlation=False)
        idx = 0
        expected_sum = 0
        for scenario in scenarios:
            _, multiplicity = ids_with_multiplicities[idx]
            idx += 1
            expected_sum += np.sum(scenario)*multiplicity
        expected_sum /= self.__no_of_validation_scenarios
        #print('Expected sum:', expected_sum)
        #print('Limit:', expected_sum_constraint.get_sum_limit())
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
                package_dict, attribute, True
            )
        
        mat = np.array(scenarios)
        mults = np.array([m for _, m in ids_with_multiplicities])
        scenario_scores = np.sort(mat.T @ mults)[::-1]
        scenarios_to_consider = \
            min(
                int(np.floor((var_constraint.get_probability_threshold()*\
                          self.__no_of_validation_scenarios))),
                self.__no_of_validation_scenarios - 1
            )
        print('Returning VaR:', scenario_scores[scenarios_to_consider])
        return float(scenario_scores[scenarios_to_consider])


    def get_var_constraint_satisfaction(
            self, package_dict,
            var_constraint: VaRConstraint
    ) -> float:
        if package_dict is None:
            #print('Null package, returning satisfies all scenarios')
            return 1.00
        attribute = var_constraint.get_attribute_name()
        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute, True
            )
        mat = np.array(scenarios)
        mults = np.array([m for _, m in ids_with_multiplicities])
        scenario_scores = mat.T @ mults
        limit = var_constraint.get_sum_limit()
        if var_constraint.get_inequality_sign() == \
                RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            satisfying_scenarios = int(np.sum(scenario_scores >= limit))
        elif var_constraint.get_inequality_sign() == \
                RelationalOperators.LESS_THAN_OR_EQUAL_TO:
            satisfying_scenarios = int(np.sum(scenario_scores <= limit))
        else:
            satisfying_scenarios = int(np.sum(scenario_scores == limit))
        return satisfying_scenarios / self.__no_of_validation_scenarios
    

    def get_cvar_among_validation_scenarios(
        self, package_dict,
        cvar_constraint: CVaRConstraint 
    ) -> float:
        if package_dict is None:
            #print('Empty package')
            if cvar_constraint.get_inequality_sign() == \
                RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                #print('Returning cvar inf')
                return math.inf
            else:
                #print('Returning cvar -inf')
                return -math.inf
        attribute = cvar_constraint.get_attribute_name()
        
        scenarios, ids_with_multiplicities = \
            self.__get_scenarios_and_ids(
                package_dict, attribute, True
            )
        mat = np.array(scenarios)
        mults = np.array([m for _, m in ids_with_multiplicities])
        scenario_scores = mat.T @ mults
        if cvar_constraint.get_tail_type() == TailType.HIGHEST:
            scenario_scores = np.sort(scenario_scores)[::-1]
        else:
            scenario_scores = np.sort(scenario_scores)

        no_of_scenarios_to_consider = \
            max(1, int(
                np.floor(
                    self.__no_of_validation_scenarios*\
                    cvar_constraint.get_percentage_of_scenarios()\
                        /100
                )
            ))

        return float(np.average(scenario_scores[
            0: no_of_scenarios_to_consider]))


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
        print('Probability of VaR:', probability)
        print('Required min threshold:',
             var_constraint.get_probability_threshold())
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
            self.get_cvar_among_validation_scenarios(
                package_dict, cvar_constraint,
            )
        
        #print('CVaR:', cvar)
        #print('Sum limit:', cvar_constraint.get_sum_limit())
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
                #print('Checking Expected Sum Constraint')
                if not self.get_expected_sum_constraint_feasibility(
                    package_dict, constraint
                ):
                    #print('Expected Sum Constraint is not satisfied')
                    return False
                #else:
                    #print('Expected sum constraint is satisfied')
            if constraint.is_var_constraint():
                #print('Checking VaR constraint')
                if not self.get_var_constraint_feasibility(
                    package_dict, constraint
                ):
                    #print('VaR constraint is not satisfied')
                    return False
                #else:
                    #print('VaR constraint satisfied')
            if constraint.is_cvar_constraint():
                if not self.get_cvar_constraint_feasibility(
                    package_dict, constraint
                ):
                    #print('CVaR constraint is not satisfied')
                    return False
                #else:
                #    print('CVaR constraint satisfied')
        return True


    def is_package_1_pm_epsilon_approximate(
        self, package_dict: dict,
        epsilon: float, upper_bound: float
    ):
        if package_dict is None:
            #print('Empty package, not optimal enough')
            return False

        objective = \
            self.__query.get_objective()
        
        objective_value = \
            self.get_validation_objective_value(
                package_dict
            )
        #print('Validation objective value:',
        #    objective_value)
        #print('Upper bound:', upper_bound)
        #print('Epsilon:', epsilon)
        if objective.get_objective_type() == \
            ObjectiveType.MAXIMIZATION:
            if upper_bound >= 0:
                return objective_value >= \
                    (1 - epsilon) * upper_bound
            return objective_value >= \
                (1 + epsilon) * upper_bound
        
        if upper_bound >= 0:
            return objective_value <= \
                (1 + epsilon) * upper_bound
        return objective_value <= \
            (1 - epsilon) * upper_bound
