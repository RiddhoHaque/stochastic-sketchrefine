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
            maxed_out_duplicate_vector
        self.__partition_id_in_duplicate_vector = \
            partition_id_in_duplicate_vector
        self.__scenario_cache = {}

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
                    assert len(self.__correlations[attribute][idx]) == self.__maxed_out_duplicate_vector[idx]
                print('Got initial correlations')

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

    def update_partition_id_in_duplicate_vector(
        self, partition_id_in_duplicate_vector: dict
    ):
        #print('Updating Partition ID in duplicate vector')
        self.__partition_id_in_duplicate_vector =\
            partition_id_in_duplicate_vector
        self.__scenario_cache = {}

    def bump_correlations_for_partitions(
        self, local_pids: list, delta: float = 0.1
    ) -> bool:
        if not self.__dbInfo.has_inter_tuple_correlations():
            return False
        any_increased = False
        for attribute in self.__correlations:
            for pid in local_pids:
                for d in range(len(self.__correlations[attribute][pid])):
                    old = self.__correlations[attribute][pid][d]
                    new_val = min(1.0, old + delta)
                    if new_val > old:
                        any_increased = True
                    self.__correlations[attribute][pid][d] = new_val
        self.__scenario_cache.clear()
        return any_increased

    def __get_scenarios_and_ids(self, package_dict: dict,
                              attribute: str,
                              consider_correlation: bool):
        cache_key = (attribute, consider_correlation,
                     tuple(sorted(package_dict.items())))
        if cache_key in self.__scenario_cache:
            return self.__scenario_cache[cache_key]
        base_predicate = ''
        pids_with_multiplicities = []
        
        for id in package_dict:
            pid = self.__partition_id_in_duplicate_vector[id]
            #print('ID in package:', id, 'Partition ID:', pid)
            did_pid_appear_before = False
            for _ in range(len(pids_with_multiplicities)):
                old_pid, old_multiplicity = pids_with_multiplicities[_]
                if old_pid == pid:
                    did_pid_appear_before = True
                    pids_with_multiplicities[_] = pid, old_multiplicity +\
                        package_dict[id]
                    break
            
            if not did_pid_appear_before:
                pids_with_multiplicities.append(
                    (pid, package_dict[id]))
                if len(base_predicate) > 0:
                    base_predicate += " or "
                base_predicate += " id=" + str(pid)
        
        pids_with_multiplicities.sort()
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
                if multiplicity < \
                    self.__maxed_out_duplicate_vector[pid]:
                    duplicates.append(int(multiplicity))
                else:
                    duplicates.append(
                        self.__maxed_out_duplicate_vector[pid])
                #print('Number of Duplicates:', duplicates[-1])
                #print('Correlations available for',
                #      len(self.__correlations[attribute][pid]),
                #      'duplicates')
                correlations.append(
                    self.__correlations[attribute][pid][duplicates[-1]-1])
                #print('Appended correlation:', correlations[-1])
                pids.append(pid)
                representatives.append(
                    self.__representatives[attribute][pid]
                )
                #print('Representative:', self.__representatives[attribute][pid])
        else:
            #print('Generating Scenarios without correlation')
            duplicates = []
            representatives = []
            for pid, multiplicity in pids_with_multiplicities:
                if multiplicity < \
                    self.__maxed_out_duplicate_vector[pid]:
                    duplicates.append(int(multiplicity))
                else:
                    duplicates.append(
                        self.__maxed_out_duplicate_vector[pid])
                representatives.append(
                    self.__representatives[attribute][pid]
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
            duplicates = self.__maxed_out_duplicate_vector[pid]
            if multiplicity < duplicates:
                duplicates = int(multiplicity)
            init_index = self.__prefix_sum_max_duplicates[pid]

            extent = multiplicity % duplicates
            distribution = int(np.floor(multiplicity/duplicates))

            #print('No. of duplicates:', duplicates)
            for duplicate in range(duplicates):
                id = init_index + duplicate
                multiplicity = distribution
                if duplicate < extent:
                    multiplicity += 1
                #print('Duplicate', duplicate,
                #      'multiplicity:', multiplicity)
                ids_with_multiplicities.append(
                    (id, multiplicity))

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
            
        attribute = \
            self.__query.get_objective().get_attribute_name()
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
            int(
                np.floor(
                    self.__no_of_validation_scenarios*\
                    cvar_constraint.get_percentage_of_scenarios()\
                        /100
                )
            )

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
            return objective_value >= \
                (1 - epsilon) * upper_bound
        
        return objective_value <= \
            (1 + epsilon) * upper_bound
