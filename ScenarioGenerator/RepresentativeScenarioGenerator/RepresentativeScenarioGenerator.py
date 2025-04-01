import time
import numpy as np
from numpy.random import SFC64, SeedSequence, Generator
from scipy.stats import norm

from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from Utils.Relation_Prefixes import Relation_Prefixes


class RepresentativeScenarioGenerator(ScenarioGenerator):

    def __init__(self, 
                 relation: str,
                 attr: str,
                 base_predicate = '',
                 duplicate_vector = [],
                 correlation_coeff = [0.01]) -> None:
        self.__relation = relation
        self.__attribute = attr
        self.__base_predicate = base_predicate
        self.__duplicate_vector = duplicate_vector
        self.__correlation_coeff = correlation_coeff

    
    def __get_partition_ids(self):
        histogram_relation = \
            Relation_Prefixes.HISTOGRAM_RELATION_PREFIX +\
                self.__relation
        sql = 'SELECT DISTINCT partition_id FROM ' +\
            histogram_relation + ' WHERE attribute=' +\
            "'" + self.__attribute + "'"
        if len(self.__base_predicate) > 0:
            sql += ' AND ' + self.__base_predicate
        sql += ' ORDER BY partition_id;'
        PgConnection.Execute(sql)
        raw_pids = PgConnection.Fetch()
        pids = []
        for tuple in raw_pids:
            pids.append(tuple[0])
        return pids
    

    def __get_histogram(self, pid: int):
        histogram_relation = \
            Relation_Prefixes.HISTOGRAM_RELATION_PREFIX +\
                self.__relation
        sql = 'SELECT bar_start, bar_width, start_cdf,' +\
            ' prob_width FROM ' + histogram_relation +\
            ' WHERE partition_id = ' + str(pid) +\
            ' AND attribute = ' + "'" + self.__attribute +\
            "'"

        if len(self.__base_predicate) > 0:
            sql += ' AND ' + self.__base_predicate
        sql += ' ORDER BY bar_start;'
        PgConnection.Execute(sql)
        return PgConnection.Fetch()     
    

    def __get_values(self, pid: int, location: dict,
                     norta_vec: list[list[float]],
                     bars):
        cdfs = sorted(location.keys())
        if bars is None:
            bars = self.__get_histogram(pid)
        cdf_index = 0
        
        for bar_start, bar_width, start_cdf, prob_width\
            in bars:
            while start_cdf + prob_width >= cdfs[cdf_index]:
                value = bar_start + (
                    (cdfs[cdf_index] -start_cdf)*bar_width)/\
                        prob_width
                for row, column in location[cdfs[cdf_index]]:

                    norta_vec[row][column] = value
                    cdf_index += 1
                    if cdf_index >= len(cdfs):
                        break
                if cdf_index >= len(cdfs):
                    break
            if cdf_index >= len(cdfs):
                break
        return norta_vec
            

    def generate_scenarios(
        self, seed: int, no_of_scenarios: int, pid=None,
        bins=None, duplicates_to_use=-1, correlation_to_use = -2
    ) -> list[list[float]]:
        scenarios = []
        rng = Generator(SFC64(SeedSequence(seed)))
        if pid is None:
            pids = self.__get_partition_ids()
        else:
            pids = [pid]
        

        location = dict()

        for _ in range(len(pids)):
            pid = pids[_]
            
            if duplicates_to_use == -1:
                duplicates = self.__duplicate_vector[_]
            else:
                duplicates = duplicates_to_use
            if correlation_to_use == -2:
                correlation = self.__correlation_coeff[_]
            else:
                correlation = correlation_to_use

            start_time = time.time()
            norta_vec = rng.standard_normal(
                size=(duplicates, no_of_scenarios))
            
            lambda_1 = np.sqrt(1 + (duplicates - 1)*\
                               correlation)
            lambda_2 = np.sqrt(1 - correlation)
            
            # Vectorized sum across the `duplicate` axis (excluding the first element)
            sum_others = np.sum(norta_vec[1:duplicates, :], axis=0)

            # Update the first row of `norta_vec` vectorized
            norta_vec[0, :] = norta_vec[0, :] * lambda_1 - sum_others * lambda_2

            # Apply the norm.cdf and update the location dictionary for the first row
            keys = norm.cdf(norta_vec[0, :])
            for scenario, key in enumerate(keys):
                if key not in location:
                    location[key] = []
                location[key].append((0, scenario))

            # Vectorized update for the rest of the duplicates
            norta_vec[1:duplicates, :] *= lambda_2
            norta_vec[1:duplicates, :] -= norta_vec[0, :] * lambda_1

            # Apply the norm.cdf and update the location dictionary for the duplicates
            keys_duplicates = norm.cdf(norta_vec[1:duplicates, :])

            for duplicate in range(1, duplicates):
                for scenario, key in enumerate(keys_duplicates[duplicate-1, :]):
                    if key not in location:
                        location[key] = []
                    location[key].append((duplicate, scenario))
            # print(time.time() - start_time, 'seconds to finish norta vec')
            start_time = time.time()
            norta_vec = self.__get_values(pid, location,
                                          norta_vec, bins)
            
            # print(time.time() - start_time, 'seconds to finish cdf matching')
            for vec in norta_vec:
                scenarios.append(vec)
            
        return scenarios
    

    def generate_scenarios_multiple_pids(
        self, seed: int, no_of_scenarios: int,
        pids: list[int], bins_list,
        duplicates: list[int],
        correlations_list: list[float]):
        scenarios = []
        for _ in range(len(pids)):
            pid = pids[_]
            bins = bins_list[_]
            new_scenarios = self.generate_scenarios(
                seed, no_of_scenarios, pid, bins,
                duplicates[_],
                correlations_list[_]
            )
            for scenario in new_scenarios:
                scenarios.append(scenario)
        return scenarios
