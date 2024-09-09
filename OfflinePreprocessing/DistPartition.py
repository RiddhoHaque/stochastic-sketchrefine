import numpy as np
from numpy.random import SFC64, SeedSequence, Generator

from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from SeedManager.SeedManager import SeedManager
from ValueGenerator.ValueGenerator import ValueGenerator


class DistPartition:

    def __init__(self, relation: str,
                 dbInfo: DbInfo):
        self.__relation = relation
        self.__dbInfo = dbInfo
        self.__size_threshold = \
            Hyperparameters.SIZE_THRESHOLD
        self.__diameter_thresholds = dict()
        self.__values = dict()
        self.__total_tuples = self.__get_number_of_tuples()
        
        self.__no_of_partitions = 0
        self.__partitioning_seed = SeedManager.get_next_seed()
        self.__tuples_whose_partitions_are_found = 0
        self.__partition_no = dict()
        self.__pivot_generator = \
            Generator(SFC64(SeedSequence(SeedManager.get_next_seed())))
        
        for det_attr in self.__dbInfo.get_deterministic_attributes():
            self.__diameter_thresholds[det_attr] = \
                self.__dbInfo.get_diameter_threshold(
                    det_attr)
            self.__values[det_attr] = \
                self.__get_values(0, self.__total_tuples-1,
                                  det_attr)
            print('Created values for', det_attr)
        
        self.__scenarios = dict()
        for stoch_attr in self.__dbInfo.get_stochastic_attributes():
            self.__diameter_thresholds[stoch_attr] = \
                self.__dbInfo.get_diameter_threshold(
                    stoch_attr)
            self.__scenarios[stoch_attr] = \
                self.__get_scenarios(0, self.__total_tuples-1,
                                     stoch_attr)
            print('Created Scenarios for', stoch_attr)


    def __get_values(
            self,
            interval_start: int,
            interval_end: int,
            attribute: str):
        return ValueGenerator(
            relation=self.__relation,
            base_predicate='id >= ' + \
                str(interval_start) + \
                ' and id <= ' + \
                str(interval_end),
            attribute=attribute
        ).get_values()
    
    
    def __get_scenarios(
            self,
            interval_start: int,
            interval_end: int,
            attribute: str):
        vg_function = \
            self.__dbInfo.get_variable_generator_function(
                attribute)
        return vg_function(
            relation = self.__relation,
            base_predicate = 'id >= ' + str(interval_start)\
                + ' and id <= ' + str(interval_end)
        ).generate_scenarios(
            seed=self.__partitioning_seed,
            no_of_scenarios = Hyperparameters.MAD_NO_OF_SAMPLES
        )

    
    def __get_number_of_tuples(self) -> int:
        sql_query = \
            "SELECT COUNT(*) FROM " + self.__relation\
                + ";"
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()[0][0]
    
    
    def __mean_absolute_relative_difference(
        self, v1: float, v2: float):
        if v1 > v2:
            if v2 > 0:
                return (v1-v2)/v2
            elif v2 == 0:
                return (v1-v2)*99999
            else:
                return -(v1-v2)/v2
        if v1 > 0:
            return (v2-v1)/v1
        elif v1 == 0:
            return (v2-v1)*99999
        return -(v2-v1)/v1
    

    def __mean_absolute_difference(
        self, v1: float, v2: float):
        if v1 > v2:
            return (v1-v2)
        return (v2-v1)

    
    def __get_distance_id_pairs(
        self, ids: list[int], pivot: int,
        attribute: str) -> list[(float, int)]:
        distance_id_pairs = []
        if self.__dbInfo.is_deterministic_attribute(
            attribute):
            for id in ids:
                distance_id_pairs.append(
                    (self.__mean_absolute_difference(
                        self.__values[attribute][id][0],
                        self.__values[attribute][pivot][0]),
                    id)
                )
        else:
            for id in ids:
                distance_id_pairs.append(
                    (np.average(
                        [self.__mean_absolute_difference(
                            self.__scenarios[attribute][id][_],
                            self.__scenarios[attribute][pivot][_])
                        for _ in range(len(
                            self.__scenarios[attribute][id]))]
                    ), id)
                )
        distance_id_pairs.sort()
        return distance_id_pairs
    
    
    def get_ids_with_increasing_distances(
            self, attribute: str, ids: list[int]):
        pivot = ids[self.__pivot_generator.integers(
            low=0, high=len(ids), size=1)[0]]
        id_distance_pairs = self.__get_distance_id_pairs(ids, pivot, 
                                                         attribute)
        _, farthest_tuple = id_distance_pairs[-1]
        id_distance_pairs = self.__get_distance_id_pairs(ids,
                                    farthest_tuple, attribute)
        return id_distance_pairs

    
    def get_scenario_values(self, attr: str, tuple_id: int):
        return self.__scenarios[attr][tuple_id]

    
    def partition(self, ids: list[int],
                  depth = 1):
        if len(ids) == 1:
            self.__partition_no[ids[0]] = self.__no_of_partitions
            self.__no_of_partitions += 1
            self.__tuples_whose_partitions_are_found += 1
            print('No. of partitions formed:', self.__no_of_partitions)
            print('Tuples whose partitions are found:', self.__tuples_whose_partitions_are_found)
            return
        
        attributes = []

        for det_attr in self.__dbInfo.get_deterministic_attributes():
            attributes.append(det_attr)
        
        for stoch_attr in self.__dbInfo.get_stochastic_attributes():
            attributes.append(stoch_attr)
        
        distances_and_ids_for_widest_attr = None
        current_highest_ratio = -1.0
        attribute_with_highest_ratio = None

        for attribute in attributes:
            #print('Pivotscanning for', attribute)
            distances_and_ids = self.get_ids_with_increasing_distances(
                    attribute, ids
                )
            #print('Pivotscan done')
            farthest_distance, _ = distances_and_ids[-1]
            #print('Farthest distance:', farthest_distance)
            if farthest_distance / self.__diameter_thresholds[
                attribute] > current_highest_ratio:
                distances_and_ids_for_widest_attr = \
                    distances_and_ids
                attribute_with_highest_ratio = attribute
                current_highest_ratio = farthest_distance / \
                    self.__diameter_thresholds[attribute]

        if len(ids) > self.__size_threshold:
            print('Size threshold exceeded at depth', depth, 
                  'with', len(ids), 'tuples')
            temp_ids = []
            for _, id in distances_and_ids_for_widest_attr:
                temp_ids.append(id)
                if len(temp_ids) == self.__size_threshold:
                    self.partition(temp_ids, depth+1)
                    temp_ids = []
            if len(temp_ids) > 0:
                self.partition(temp_ids, depth+1)
            return
    
        if current_highest_ratio > 1.0:
            print('Diameter threshold exceeded for ', attribute_with_highest_ratio,
                  'with ratio', current_highest_ratio, 'at depth', depth, 
                  'with', len(ids), 'tuples')
            temp_ids = []
            multiple = 1
            _, pivot = distances_and_ids_for_widest_attr[0]
            for distance, id in distances_and_ids_for_widest_attr:
                if distance > multiple*self.__diameter_thresholds[attribute_with_highest_ratio]:
                    self.partition(temp_ids, depth+1)
                    while distance > multiple*self.__diameter_thresholds[
                        attribute_with_highest_ratio]:
                        multiple += 1
                    temp_ids = []
                temp_ids.append(id)
            
            if len(temp_ids) > 0:
                self.partition(temp_ids, depth+1)
            return

        for id in ids:
            self.__partition_no[id] = self.__no_of_partitions
        self.__no_of_partitions += 1
        self.__tuples_whose_partitions_are_found += len(ids)
        print('No. of partitions formed:', self.__no_of_partitions)
        print('Tuples whose partitions are found:', self.__tuples_whose_partitions_are_found)

    
    def get_no_of_partitions(self):
        return self.__no_of_partitions