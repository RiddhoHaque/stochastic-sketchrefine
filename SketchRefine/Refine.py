import numpy as np

from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from StochasticPackageQuery.Query import Query
from Utils.Stochasticity import Stochasticity
from Utils.Relation_Prefixes import Relation_Prefixes
from ValueGenerator.RepresentativeValueGenerator import RepresentativeValueGenerator
from ValueGenerator.ValueGenerator import ValueGenerator



class Refine:

    def __init__(
        self, partition_groups: list[list[int]],
        no_of_optimization_scenarios: int,
        max_no_of_duplicates: list[int],
        sketch_objective_value: float,
        sketch_package: dict[int, int],
        query: Query, dbInfo: DbInfo):

        self.__partition_groups = partition_groups
        self.__no_of_optimization_scenarios =\
            no_of_optimization_scenarios
        self.__no_of_validation_scenarios =\
            Hyperparameters.NO_OF_VALIDATION_SCENARIOS
        self.__max_no_of_duplicates =\
            max_no_of_duplicates
        self.__sketch_objective_value =\
            sketch_objective_value
        self.__sketch_package =\
            sketch_package
        
        self.__query = query
        self.__tuples_in_partition =\
            dict()
        self.get_tuples_in_each_partition()
        
        self.__representatives =\
            dict()
        self.get_representative_of_each_partition()

        self.__stochastic_attributes = \
            self.__get_stochastic_attributes()
        self.__scenarios = dict()
        self.__values = dict()
        self.__dbInfo = dbInfo


    def get_tuples_in_each_partition(self):    
        base_predicate = ''
        for partition_group in self.__partition_groups:
            for partition_id in partition_group:
                if len(base_predicate) > 0:
                    base_predicate += ' or '
                base_predicate += 'partition_id = ' +\
                    str(partition_id)
        
        sql = 'SELECT partition_id, tuple_id FROM ' +\
            Relation_Prefixes.PARTITION_RELATION_PREFIX +\
            self.__query.get_relation() + ' WHERE ' +\
            base_predicate + ' ORDER BY (partition_id, tuple_id);'
        
        PgConnection.Execute(sql)

        tuples = PgConnection.Fetch()
        
        for pid, tid in tuples:
            if pid not in self.__tuples_in_partition:
                self.__tuples_in_partition[pid] =\
                    []
            self.__tuples_in_partition[pid].append(
                tid)


    def get_representative_of_each_partition(self):
        base_predicate = ''
        for partition_group in self.__partition_groups:
            for partition_id in partition_group:
                if len(base_predicate) > 0:
                    base_predicate += ' or '
                base_predicate += 'partition_id = ' +\
                    str(partition_id)
        
        sql = 'SELECT partition_id, attribute, representative_tuple_id ' +\
            'FROM ' + Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX +\
            self.__query.get_relation() + ' WHERE ' + base_predicate + ';'
        
        PgConnection.Execute(sql)
        tuples = PgConnection.Fetch()

        for partition_id, attribute, representative_tuple_id in tuples:
            self.__representatives[(partition_id, attribute)] =\
                representative_tuple_id


    def __get_stochastic_attributes(self):
        attributes = set()
        for constraint in self.__query.get_constraints():
            if constraint.is_expected_sum_constraint():
                attributes.add(
                    constraint.get_attribute_name())
            if constraint.is_risk_constraint():
                attr = constraint.get_attribute_name()
                attributes.add(attr)
        
        if self.__query.get_objective().get_stochasticity()\
            == Stochasticity.STOCHASTIC:
            attributes.add(self.__query.get_objective().\
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


    def solve(self):
        forward_bins = []        
        self.__partition_groups.reverse()

        for partition_group in self.__partition_groups:
            forward_bins.append(partition_group)

        chosen_tuples_per_bin = []

        group_index = 0

        while group_index < len(forward_bins):
            chosen_tuples_with_multiplicity = []
            for previous_index in range(0, group_index-1):
                for chosen_tuple, multiplicity in chosen_tuples_per_bin[previous_index]:
                    chosen_tuples_with_multiplicity.append(
                        (chosen_tuple, multiplicity))

            remaining_partitions_with_multiplicty = []

            for next_index in range(group_index+1, len(forward_bins)):
                for pid in forward_bins[next_index]:
                    remaining_partitions_with_multiplicty.append(
                        (pid, int(self.__sketch_package[pid])))


            package, objective_value = \
                self.solve_partition(
                    forward_bins[group_index],
                    chosen_tuples_with_multiplicity,
                    remaining_partitions_with_multiplicty
                )
            
            if package is None:
                if group_index > 0:
                    forward_bins[group_index], forward_bins[group_index-1] =\
                        forward_bins[group_index-1], forward_bins[group_index]
                    chosen_tuples_per_bin.pop()
                    group_index -= 1
                else:
                    print('Sketch package could not be refined')
            
            else:
                number_of_tuples_in_group = 0
                for pid in forward_bins[group_index]:
                    number_of_tuples_in_group += \
                        len(self.__tuples_in_partition[pid])
                
                tuple_ids = []

                for id in package.keys():
                    if id < number_of_tuples_in_group:
                        tuple_ids.append(id)
                
                tuple_ids.sort()

                pid_index = 0
                sum = 0
                chosen_tuples_per_bin.append([])

                for tuple_id in tuple_ids:
                    while tuple_id >= sum +\
                        len(self.__tuples_in_partition[
                            forward_bins[group_index][pid_index]]):
                        sum += len(self.__tuples_in_partition[
                            forward_bins[group_index][pid_index]])
                        pid_index += 1
                    
                    id = self.__tuples_in_partition[
                        forward_bins[group_index][pid_index]][tuple_id - sum]
                    chosen_tuples_per_bin[-1].append((id, package[tuple_id]))

                group_index += 1
        
        refined_package = dict()

        for chosen_tuples_in_bin in chosen_tuples_per_bin:
            for chosen_tuple, multiplicity in chosen_tuples_in_bin:
                refined_package[chosen_tuple] = multiplicity
        
        return refined_package, objective_value


    def solve_partition(
        self, 
        partition_group,
        chosen_tuples_with_multiplicity,
        remaining_partitions_with_multiplicity):

        sizes = []
        for attribute in self.__stochastic_attributes:
            self.__scenarios[attribute] = []
            
            for partition_id in partition_group:
                partition_wise_scenario_generator =\
                self.__dbInfo.get_variable_generator_function(
                    attribute
                )(
                    relation=self.__query.get_relation(),
                    base_predicate=self.__query.get_base_predicate()
                )
                
                scenarios = partition_wise_scenario_generator.\
                    generate_scenarios_from_partition(
                        seed=Hyperparameters.INIT_SEED,
                        no_of_scenarios=self.__no_of_optimization_scenarios,
                        partition_id=partition_id
                    )
                
                sizes.append(len(scenarios))
                
                for scenario in scenarios:
                    self.__scenarios[attribute].append(scenario)
            
            for tuple_id, _ in chosen_tuples_with_multiplicity:
                tuple_wise_scenario_generator =\
                    self.__dbInfo.get_variable_generator_function(
                        attribute)(
                        relation=self.__query.get_relation(),
                        base_predicate='id=' + str(tuple_id)
                    )
                
                scenarios = tuple_wise_scenario_generator.\
                    generate_scenarios(
                        seed=Hyperparameters.INIT_SEED,
                        no_of_scenarios=self.__no_of_optimization_scenarios
                    )
                
                sizes.append(len(scenarios))
                
                for scenario in scenarios:
                    self.__scenarios[attribute].append(scenario)
                
            for partition_id, multiplicity in remaining_partitions_with_multiplicity:
                representative_id = self.__representatives[
                        (partition_id, attribute)]
                    
                no_of_duplicates = self.__max_no_of_duplicates[
                    partition_id]
                    
                if multiplicity < no_of_duplicates:
                    no_of_duplicates = multiplicity
                    
                representative_scenario_generator =\
                    self.__dbInfo.get_variable_generator_function(
                        attribute)(
                        relation=self.__query.get_relation(),
                        base_predicate='id=' + str(representative_id)
                    )
                
                scenarios = representative_scenario_generator.\
                    generate_scenarios(
                        seed=Hyperparameters.INIT_SEED,
                        no_of_scenarios=(self.__no_of_optimization_scenarios*\
                            no_of_duplicates)
                    )
                    
                scenarios = np.reshape(scenarios,\
                                        (no_of_duplicates,
                                        self.__no_of_optimization_scenarios))
                    
                sizes.append(no_of_duplicates)

                for scenario in scenarios:
                    self.__scenarios[attribute].append(scenario)

        for attribute in self.__get_deterministic_attributes():
            self.__values[attribute] = []
            
            for partition_id in partition_group:
                value_generator = ValueGenerator(
                    relation=self.__query.get_relation(),
                    base_predicate=self.__query.get_base_predicate(),
                    attribute=attribute
                )
                values = value_generator.get_values_from_partition(
                    partition_id
                )

                for value in values:
                    self.__values[attribute].append(value)
            
            for tuple_id, _ in chosen_tuples_with_multiplicity:
                value_generator =\
                    ValueGenerator(
                        relation=self.__query.get_relation(),
                        base_predicate='id=' + str(tuple_id),
                        attribute=attribute
                    )
                
                values = value_generator.get_values_from_partition()

                for value in values:
                    self.__values[attribute].append(value)
            
            for partition_id, multiplicity in remaining_partitions_with_multiplicity:
                representative_id = self.__representatives[(partition_id, attribute)]
                    
                no_of_duplicates = self.__max_no_of_duplicates[partition_id]

                if multiplicity < no_of_duplicates:
                    no_of_duplicates = multiplicity

                representative_value_generator =\
                    RepresentativeValueGenerator(
                        relation=self.__query.get_relation(),
                        base_predicate='partition_id='+str(partition_id),
                        attribute=attribute,
                        duplicate_vector=[no_of_duplicates]
                    )
                    
                values = representative_value_generator.get_values()

                for value in values:
                    self.__values[attribute].append(value)
        
