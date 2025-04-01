import time
import numpy as np

from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGenerator import RepresentativeScenarioGenerator
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGeneratorWithoutCorrelation import RepresentativeScenarioGeneratorWithoutCorrelation
from SeedManager.SeedManager import SeedManager
from StochasticPackageQuery.Query import Query
from SketchRefine.SketchRCLSolve import SketchRCLSolve
from SketchRefine.SketchValidator import SketchValidator
from Utils.Relation_Prefixes import Relation_Prefixes
from Utils.Stochasticity import Stochasticity
from ValueGenerator.RepresentativeValueGenerator import RepresentativeValueGenerator


class Sketch:

    def __init__(
        self, query: Query, dbInfo: DbInfo,
        no_of_opt_scenarios: int,
        is_lp_relaxation = False
    ):
        self.__query = query
        self.__dbInfo = dbInfo
        self.__no_of_opt_scenarios = \
            no_of_opt_scenarios
        self.__partition_ids = \
            self.__get_no_of_partitions()
        self.__max_no_of_duplicates,\
            self.__partition_sizes = \
            self.__get_max_no_of_duplicates()
        
        self.__scenarios = dict()
        self.__values = dict()
        self.__constraints_for_attribute = dict()
        self.__stochastic_attributes = \
            self.__get_stochastic_attributes()
        
        for attribute in self.__stochastic_attributes:
            self.__scenarios[attribute] = \
                RepresentativeScenarioGeneratorWithoutCorrelation(
                    relation=self.__query.get_relation(),
                    attr=attribute,
                    scenario_generator=\
                        self.__dbInfo.\
                            get_variable_generator_function(
                                attribute),
                    duplicates=self.__max_no_of_duplicates
                ).generate_scenarios(
                    seed=Hyperparameters.INIT_SEED,
                    no_of_scenarios=self.__no_of_opt_scenarios
                )
            
        
        self.__alpha, self.__duplicate_vector = \
            self.__get_init_duplicate_vector()
        
        self.__clip_scenarios()
        
        self.__partition_id_in_duplicate_vector = \
            dict()
        
        tid = 0
        pid = 0
        for value in self.__duplicate_vector:
            for _ in range(value):
                self.__partition_id_in_duplicate_vector[tid] = pid
                tid += 1
            pid += 1

        for attribute in self.__get_deterministic_attributes():
            self.__values[attribute] = \
                RepresentativeValueGenerator(
                    relation=self.__query.get_relation(),
                    base_predicate='', attribute=attribute,
                    duplicate_vector=self.__duplicate_vector
                ).get_values()

        self.__is_lp_relaxation = is_lp_relaxation
        self.__validator = SketchValidator(
            query=self.__query,
            dbInfo=self.__dbInfo,
            no_of_validation_scenarios=\
                Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
            maxed_out_duplicate_vector=\
                self.__max_no_of_duplicates,
            partition_id_in_duplicate_vector=\
                self.__partition_id_in_duplicate_vector
        )
        
        self.__solver = SketchRCLSolve(
            query=self.__query,
            linear_relaxation=self.__is_lp_relaxation,
            scenarios=self.__scenarios,
            values=self.__values,
            no_of_variables=np.sum(self.__duplicate_vector),
            no_of_opt_scenarios=self.__no_of_opt_scenarios,
            validator=self.__validator,
            approximation_bound=Hyperparameters.APPROXIMATION_BOUND,
            sampling_tolerance=Hyperparameters.SAMPLING_TOLERANCE,
            bisection_threshold=Hyperparameters.BISECTION_THRESHOLD,
            partition_sizes=self.__partition_sizes
        )

    def __get_no_of_partitions(self) -> list[int]:
        sql = "SELECT distinct partition_id FROM " +\
                Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX +\
                    self.__query.get_relation() +\
            " ORDER BY partition_id;"
        
        PgConnection.Execute(sql)
        tuples = PgConnection.Fetch()

        partition_ids = []
        for tuple in tuples:
            partition_ids.append(tuple[0])
        
        return partition_ids
    

    def __get_max_no_of_duplicates(self) -> list[int]:
        sql = "SELECT COUNT(*) FROM " +\
                Relation_Prefixes.PARTITION_RELATION_PREFIX +\
                    self.__query.get_relation() +\
            " GROUP BY partition_id ORDER BY partition_id;"
        
        PgConnection.Execute(sql)
        tuples = PgConnection.Fetch()

        max_no_of_duplicates = []
        partition_sizes = []

        for tuple in tuples:
            if tuple[0] < Hyperparameters.MAX_TUPLES_IN_PACKAGE:
                max_no_of_duplicates.append(tuple[0])
            else:
                max_no_of_duplicates.append(
                    Hyperparameters.MAX_TUPLES_IN_PACKAGE
                )
            partition_sizes.append(tuple[0])
        
        return max_no_of_duplicates, partition_sizes


    def __get_stochastic_attributes(self):
        attributes = set()
        for constraint in self.__query.get_constraints():
            if constraint.is_expected_sum_constraint():
                attributes.add(
                    constraint.get_attribute_name())
            if constraint.is_risk_constraint():
                attr = constraint.get_attribute_name()
                if attr not in self.__constraints_for_attribute:
                    self.__constraints_for_attribute[attr] = []
                self.__constraints_for_attribute[
                    attr].append(constraint)
                attributes.add(
                    constraint.get_attribute_name())
        
        if self.__query.get_objective().get_stochasticity() \
            == Stochasticity.STOCHASTIC:
            attributes.add(
                self.__query.get_objective().\
                    get_attribute_name())
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
                    get_attribute_name())
        return attributes
    

    def __get_duplicate_vector(self, alpha: float):
        duplicates_needed_for_partition = [
            1 for _ in range(len(self.__partition_ids))]
        
        for attribute in self.__stochastic_attributes:
            prior_duplicates = 0
            total_duplicates_needed = 0

            for partition in range(len(self.__partition_ids)):
                max_duplicates = self.__max_no_of_duplicates[partition]
                for constraint in self.__constraints_for_attribute[attribute]:
                    if constraint.is_var_constraint():
                        probability = (1-constraint.get_probability_threshold())
                    elif constraint.is_cvar_constraint():
                        probability = (1-constraint.get_percentage_of_scenarios())/100.0

                    no_of_scenarios_to_consider = int(np.floor(probability *\
                        self.__no_of_opt_scenarios))
                    
                    cumulative_lcvar_sum = [0.0]

                    for duplicates in range(1, max_duplicates+1):
                        lcvar = \
                            np.average(np.sort(self.__scenarios[
                                attribute][prior_duplicates + duplicates - 1])[
                                    :no_of_scenarios_to_consider])
                        cumulative_lcvar_sum.append(
                            cumulative_lcvar_sum[-1] + lcvar
                        )

                    gold_standard = cumulative_lcvar_sum[-1] * 1.0 / duplicates
                    if gold_standard < 1e-9:
                        gold_standard *= -1

                    for duplicates in range(1, max_duplicates+1):
                        lcvar_diff = \
                            (cumulative_lcvar_sum[duplicates]*1.0/duplicates) -\
                            gold_standard
                        
                        if lcvar_diff < 0:
                            lcvar_diff *= -1
                        
                        if gold_standard > 1e-9:
                            lcvar_diff /= gold_standard

                        if lcvar_diff <= alpha + 1e-9:
                            if duplicates > duplicates_needed_for_partition[
                                partition]:
                                duplicates_needed_for_partition[partition] =\
                                    duplicates
                                break
                    
                prior_duplicates += max_duplicates
                total_duplicates_needed += duplicates_needed_for_partition[partition]

        return total_duplicates_needed, duplicates_needed_for_partition
    

    def __get_init_duplicate_vector(self) -> list[int]:
        low_alpha = 0.0
        high_alpha = 1.0

        target = Hyperparameters.SIZE_THRESHOLD

        total_duplicates_needed, duplicate_vector = \
                self.__get_duplicate_vector(low_alpha)
        
        if total_duplicates_needed <= target:
            return low_alpha, duplicate_vector

        best_duplicate_vector = None
        best_alpha = None

        while low_alpha < high_alpha - 1e-2:
            mid_alpha = (low_alpha + high_alpha)/2.0
            total_duplicates_needed, duplicate_vector = \
                self.__get_duplicate_vector(mid_alpha)
            
            if total_duplicates_needed > target:
                low_alpha = mid_alpha
            
            else:
                high_alpha = mid_alpha
                best_alpha = mid_alpha
                best_duplicate_vector = duplicate_vector

        return best_alpha, best_duplicate_vector
                

    def __clip_scenarios(self):
        for attribute in self.__stochastic_attributes:
            prior_duplicates = 0
            scenarios = []
            for partition in range(len(self.__max_no_of_duplicates)):
                max_duplicates = self.__max_no_of_duplicates[partition]
                needed_duplicates = self.__duplicate_vector[partition]

                for duplicate_index in range(1, needed_duplicates+1):
                    scenarios.append(self.__scenarios[attribute][
                        prior_duplicates + duplicate_index - 1])
                
                prior_duplicates += max_duplicates

            self.__scenarios[attribute] = scenarios

    
    def get_partition_sizes(self):
        return self.__partition_sizes
    

    def get_max_no_of_duplicates(self):
        return self.__max_no_of_duplicates

    def solve(self):
        while self.__no_of_opt_scenarios <= \
            Hyperparameters.NO_OF_VALIDATION_SCENARIOS:

            package_dict, objective_value, needs_more_scenarios =\
                self.__solver.solve(
                    can_add_scenarios=\
                        (self.__no_of_opt_scenarios <\
                            Hyperparameters.NO_OF_VALIDATION_SCENARIOS)
                )
            
            if needs_more_scenarios:
                if self.__alpha > 1e-5:
                    self.__alpha -= Hyperparameters.ALPHA_REDUCTION
                    if self.__alpha < 0:
                        self.__alpha = 0
                elif self.__no_of_opt_scenarios < \
                    Hyperparameters.NO_OF_VALIDATION_SCENARIOS:
                    
                    previous_opt_scenarios = self.__no_of_opt_scenarios
                    self.__no_of_opt_scenarios *= 2
                    if self.__no_of_opt_scenarios >\
                        Hyperparameters.NO_OF_VALIDATION_SCENARIOS:
                        self.__no_of_opt_scenarios =\
                            Hyperparameters.NO_OF_VALIDATION_SCENARIOS
                    
                    required_optimization_scenarios = \
                        self.__no_of_opt_scenarios - previous_opt_scenarios
                    
                    for attribute in self.__stochastic_attributes:
                        new_scenarios = \
                            RepresentativeScenarioGeneratorWithoutCorrelation(
                                relation=self.__query.get_relation(),
                                attr=attribute,
                                scenario_generator=\
                                    self.__dbInfo.\
                                        get_variable_generator_function(
                                            attribute),
                                duplicates=self.__duplicate_vector
                            ).generate_scenarios(
                                seed=SeedManager.get_next_seed(),
                                no_of_scenarios=required_optimization_scenarios
                            )
                        
                        tuple_index = 0
                        for tuplewise_new_scenario in new_scenarios:
                            for value in tuplewise_new_scenario:
                                self.__scenarios[tuple_index].append(
                                    value)
                            tuple_index += 1
            else:
                break
        
        partition_package_dict = dict()
        for id in package_dict:
            pid = self.__partition_id_in_duplicate_vector[id]
            partition_package_dict[pid] = package_dict[id]
        return partition_package_dict, objective_value,\
            self.__no_of_opt_scenarios