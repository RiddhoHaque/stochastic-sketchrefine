import time
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.random import SFC64, SeedSequence, Generator
from scipy.stats.stats import pearsonr

from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OptimizationMetrics.OfflinePreprocessingMetrics import OfflinePreprocessingMetrics
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGenerator import RepresentativeScenarioGenerator
from SeedManager.SeedManager import SeedManager
from Utils.Relation_Prefixes import Relation_Prefixes
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
        self.__metrics = OfflinePreprocessingMetrics()
        
        self.__metrics.start_scenario_generation()
        for det_attr in self.__dbInfo.get_deterministic_attributes():
            self.__diameter_thresholds[det_attr] = \
                self.__dbInfo.get_diameter_threshold(
                    det_attr)
            values =\
                self.__get_values(0, self.__total_tuples-1,
                                  det_attr)
            self.__values[det_attr] = []
            for value in values:
                self.__values[det_attr].append(value[0])
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
        self.__metrics.end_scenario_generation()
        self.__ids_in_partition = []
        self.__hist_bins = dict()


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
            attribute: str,
            no_of_scenarios = Hyperparameters.MAD_NO_OF_SAMPLES):
        vg_function = \
            self.__dbInfo.get_variable_generator_function(
                attribute)
        return vg_function(
            relation = self.__relation,
            base_predicate = 'id >= ' + str(interval_start)\
                + ' and id <= ' + str(interval_end)
        ).generate_scenarios(
            seed=self.__partitioning_seed,
            no_of_scenarios = no_of_scenarios
        )

    
    def __get_number_of_tuples(self) -> int:
        sql_query = \
            "SELECT COUNT(*) FROM " + self.__relation\
                + ";"
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()[0][0]
    
    
    def __mean_absolute_difference(
        self, v1: list[float], v2: list[float]):
        abs_diff = np.abs(np.subtract(v1, v2))
        return np.average(abs_diff)
    
    def __mean_absolute_difference_accurate(
        self, v1: list[float], v2: list[float],
        tid1: int, tid2: int, attr: str):
        while True:
            abs_diff = np.abs(np.subtract(v1, v2))
            mean_abs_diff = np.average(abs_diff)
            std_abs_diff = np.std(abs_diff)
            if mean_abs_diff - 1.04*std_abs_diff > \
                self.__diameter_thresholds[attr]/2:
                return mean_abs_diff
            if mean_abs_diff + 1.04*std_abs_diff < \
                self.__diameter_thresholds[attr]/2:
                return mean_abs_diff
            if len(v1) == Hyperparameters.NO_OF_VALIDATION_SCENARIOS:
                return mean_abs_diff
        
            curr_samples = len(v1)
        
            diff = mean_abs_diff - self.__diameter_thresholds[attr]/2
            diff = 1.04*std_abs_diff/diff
            if diff < 0:
                diff *= -1
        
            req_samples = int(np.ceil(diff*diff*curr_samples))
            if req_samples > Hyperparameters.NO_OF_VALIDATION_SCENARIOS:
                req_samples = Hyperparameters.NO_OF_VALIDATION_SCENARIOS
        
            if req_samples <= curr_samples:
                return mean_abs_diff
        
            new_v1 = self.__get_scenarios(tid1, tid1, attr,
                                          req_samples - curr_samples)
            new_v2 = self.__get_scenarios(tid2, tid2, attr,
                                          req_samples - curr_samples)
                
            for v in new_v1[0]:
                v1.append(v)
            for v in new_v2[0]:
                v2.append(v) 

    
    def __get_distance_id_pairs(
        self, ids: list[int], pivot: int,
        attribute: str) -> list[(float, int)]:
        distance_id_pairs = []
        if self.__dbInfo.is_deterministic_attribute(
            attribute):
            for id in ids:
                distance = self.__values[attribute][id] - \
                    self.__values[attribute][pivot]
                if distance < 0:
                    distance *= -1
                distance_id_pairs.append(
                    (distance, id)
                )
        else:
            for id in ids:
                distance_id_pairs.append((
                    self.__mean_absolute_difference(
                        self.__scenarios[attribute][id],
                        self.__scenarios[attribute][pivot]
                    ), id))
                
        distance_id_pairs.sort()
        return distance_id_pairs
    
    def __get_distance_id_pairs_accurate(
        self, ids: list[int], pivot: int,
        attribute: str) -> list[(float, int)]:
        distance_id_pairs = []
        if self.__dbInfo.is_deterministic_attribute(
            attribute):
            for id in ids:
                distance = self.__values[attribute][id] - \
                    self.__values[attribute][pivot]
                if distance < 0:
                    distance *= -1
                distance_id_pairs.append(
                    (distance, id)
                )
        else:
            for id in ids:
                distance_id_pairs.append((
                    self.__mean_absolute_difference_accurate(
                        self.__scenarios[attribute][id],
                        self.__scenarios[attribute][pivot],
                        id, pivot, attribute
                    ), id))
                if len(self.__scenarios[attribute][id]) >\
                    Hyperparameters.MAD_NO_OF_SAMPLES:
                    self.__scenarios[attribute][id] = \
                        self.__scenarios[attribute][id][
                            :Hyperparameters.MAD_NO_OF_SAMPLES
                        ]
                if len(self.__scenarios[attribute][pivot]) >\
                    Hyperparameters.MAD_NO_OF_SAMPLES:
                    self.__scenarios[attribute][pivot] = \
                        self.__scenarios[attribute][pivot][
                            :Hyperparameters.MAD_NO_OF_SAMPLES
                        ]
                
                
        distance_id_pairs.sort()
        return distance_id_pairs
    

    def get_ids_with_increasing_distances_random_pivot(
        self, attribute: str, ids: list[int]
    ):
        pivot = ids[self.__pivot_generator.integers(
            low=0, high=len(ids), size=1)[0]]
        id_distance_pairs = \
            self.__get_distance_id_pairs(
            ids, pivot, attribute)
        return id_distance_pairs
    
    
    def get_ids_with_increasing_distances_with_pivot(
        self, attribute: str, ids: list[int], pivot: int
    ):
        id_distance_pairs = self.__get_distance_id_pairs(
            ids, pivot, attribute)
        return id_distance_pairs

    
    def get_scenario_values(self, attr: str, tuple_id: int):
        return self.__scenarios[attr][tuple_id]

    
    def partition_relation(self):
        id_lists = ValueGenerator(
            relation=self.__relation,
            base_predicate='',
            attribute='id'
        ).get_values()
    
        ids = []

        for tuple in id_lists:
            ids.append(int(tuple[0]))

        self.__metrics.start_partitioning()
        self.__partition(ids)
        self.__metrics.end_partitioning()
        print('Number of partitions:', self.get_no_of_partitions())
        self.__metrics.start_partitioning_table_creation()
        self.__form_partition_table()
        self.__metrics.end_partitioning_table_creation()
        self.__metrics.start_representative_selection()
        representatives = self.get_partition_representatives()
        print('Representatives selected')
        self.__metrics.end_representative_selection()
        if self.__dbInfo.has_inter_tuple_correlations():
            self.__metrics.start_representative_histogram_estimation()
            self.__compute_histograms(representatives)
            print('Histograms computed')
            self.__metrics.end_representative_histogram_estimation()
            self.__metrics.start_required_correlation_estimation()
            self.__compute_required_correlations(representatives)
            print('Correlations computed')
            self.__metrics.end_required_correlation_estimation()  


    def get_metrics(self):
        return self.__metrics
    

    def __plot_scatter(self, filename, vec):
        x = []
        y = []
        plt.clf()
        plt.ylabel('Runtime')
        for l in vec:
            x.append(l[0])
            y.append(l[1])
        plt.scatter(x, y)
        plt.savefig(filename)

    
    def __compute_required_correlations(self, representatives):
        
        delete_table_sql = 'DROP TABLE IF EXISTS ' +\
            Relation_Prefixes.INIT_CORRELATION_PREFIX +\
            self.__relation
        
        PgConnection.Execute(delete_table_sql)

        create_table_sql = 'CREATE TABLE ' +\
            Relation_Prefixes.INIT_CORRELATION_PREFIX +\
            self.__relation + '( partition_id INT, ' +\
            'attribute VARCHAR(25), duplicates INT, '+\
            'init_corr FLOAT);'
        
        PgConnection.Execute(create_table_sql)

        insert_sql = 'INSERT INTO ' +\
            Relation_Prefixes.INIT_CORRELATION_PREFIX +\
            self.__relation + '(partition_id, attribute, '+\
            'duplicates, init_corr) VALUES '
        
        is_first = True
        
        for pid, attr in representatives:
            if attr not in self.__dbInfo.get_stochastic_attributes():
                continue
            
            representative_id = representatives[(pid, attr)]
            partition_scenarios = []
            index = 0
            representative_index = None
            
            if not self.__dbInfo.has_inter_tuple_correlations():
                req_corr = 0.0
            else:
                for id in self.__ids_in_partition[pid]:
                    if id == representative_id:
                        representative_index = index 
                    partition_scenarios.append(
                        self.__scenarios[attr][id])
                    index += 1
                    if index == 500: # From the DKW inequality
                        break
            
                if representative_index is None:
                    partition_scenarios.append(
                        self.__scenarios[attr][representative_id])
                    representative_index = index
            
                partition_scenarios = np.subtract(
                    partition_scenarios,
                    np.mean(partition_scenarios, axis=1, keepdims=True))
            
                numerator = np.dot(partition_scenarios,
                                partition_scenarios[
                                    representative_index])
            
                med_cov_index = np.argsort(numerator)[len(numerator)//2]
                req_corr = numerator[med_cov_index] /\
                    (np.linalg.norm(partition_scenarios[representative_index]) *\
                        np.linalg.norm(partition_scenarios[med_cov_index]))

            max_dups = len(self.__ids_in_partition[pid])

            if Hyperparameters.MAX_TUPLES_IN_PACKAGE < max_dups:
                max_dups = Hyperparameters.MAX_TUPLES_IN_PACKAGE
            
            for dups in range(1, max_dups+1):
                if dups == 1:
                    mid_z_coeff = 1.0
                else:
                    lowest_z_coeff = -1.0/(dups-1) + 1e-6
                    highest_z_coeff = 1.0
                    req_corr_sum = req_corr * dups * (dups -1) + dups
                    while highest_z_coeff >\
                        lowest_z_coeff + 0.01:
                        mid_z_coeff = (lowest_z_coeff + \
                                    highest_z_coeff)/2.0
                    
                        scenarios = RepresentativeScenarioGenerator(
                            relation=self.__relation,
                            attr=attr,
                            base_predicate='partition_id=' + str(pid),
                            duplicate_vector=[dups],
                            correlation_coeff=[mid_z_coeff]
                        ).generate_scenarios(
                            seed=self.__partitioning_seed,
                            no_of_scenarios=\
                            Hyperparameters.MAD_NO_OF_SAMPLES,
                            pid=pid, bins=self.__hist_bins[pid, attr])
                        
                    
                        corr_coeff_sum = np.sum(np.corrcoef(
                            scenarios, rowvar=True))
                        
                        if corr_coeff_sum > req_corr_sum:
                            highest_z_coeff = mid_z_coeff
                        elif corr_coeff_sum < req_corr_sum:
                            lowest_z_coeff = mid_z_coeff
                        else:
                            break
                
                if is_first:
                    is_first = False
                else:
                    insert_sql += ', '
                
                insert_sql += ' (' + str(pid) + ", '" + attr +\
                    "', " + str(dups) + ', ' + str(mid_z_coeff) +\
                    ') '
                
        
        insert_sql +=";"
        PgConnection.Execute(insert_sql)
        PgConnection.Commit()


    def __compute_histograms(self, representatives):
        histogram_relation = Relation_Prefixes.HISTOGRAM_RELATION_PREFIX +\
            self.__relation
        
        delete_table_sql = 'DROP TABLE IF EXISTS ' + histogram_relation
        
        create_table_sql = 'CREATE TABLE ' + histogram_relation\
            + ' (partition_id int not null, '\
            + 'attribute varchar(25) not null, '\
            + 'bar_start double precision not null, '\
            + 'bar_width double precision not null, '\
            + 'prob_width double precision not null, '\
            + 'start_cdf double precision not null, '\
            + 'PRIMARY KEY(partition_id, attribute, bar_start, bar_width));'
        
        insert_sql = 'INSERT INTO ' + histogram_relation\
            + ' (partition_id, attribute, bar_start, bar_width,'\
            + ' start_cdf, prob_width) VALUES '
        
        is_first = True
        
        # Compute histogram for tid.attr according to
        # the algorithm in Sec 3.1.2 in the paper
        # 'Near-Optimal Density Estimation in Near-Linear
        # Time Using Variable-Width Histograms' 
                
        k = Hyperparameters.NO_OF_HIST_BINS
        eps = Hyperparameters.HIST_ERROR
        eps_prime = eps/np.log(1/eps)
        num_samples = int(np.ceil(
                    k / (eps_prime*eps_prime)))
        
        '''
        bulk_infos = []
        counter = 0
        bulk_sample_size = 50000
        bulk_info_str = dict()
        tid_lists = []
        tid_dict = dict()

        for pid, attr in representatives:
            if attr in self.__dbInfo.get_stochastic_attributes():
                tid = representatives[(pid, attr)]
                if attr in bulk_info_str:
                    bulk_info_str[attr] += ' or '
                    bulk_info_str[attr] += 'id=' + str(tid)
                    tid_dict[attr].append(tid)
                else:
                    bulk_info_str[attr] = 'id=' + str(tid)
                    tid_dict[attr] = [tid]

                counter += 1

                if counter == bulk_sample_size:
                    bulk_infos.append(bulk_info_str)
                    bulk_info_str = dict()
                    tid_lists.append(tid_dict)
                    tid_dict = dict()
                    counter = 0
        
        if len(bulk_info_str) > 0:
            bulk_infos.append(bulk_info_str)
            tid_lists.append(tid_dict)

        print('len(tid lists):', len(tid_lists))
        
        counter = 0
        bulk_samples = dict()
        bulk_info_index = 0
        '''
        for pid, attr in representatives:
            if attr in self.__dbInfo.get_stochastic_attributes():
                tid = representatives[(pid, attr)]
                
                # Doing step 2 before step 1 for efficiency

                # Step 2: Generate and normalize samples
                '''
                if counter == 0:
                    bulk_info_str = bulk_infos[bulk_info_index]
                    tid_dict = tid_lists[bulk_info_index]
                    tids = dict()
                    indices = dict()
                    for bulk_attr in tid_dict:
                        tids[bulk_attr] = tid_dict[bulk_attr]
                        tids[bulk_attr].sort()
                        indices[bulk_attr] = dict()
                        for _ in range(len(tids[bulk_attr])):
                            indices[bulk_attr][tids[bulk_attr][_]] = _
                    
                    bulk_info_index += 1
                    
                    for bulk_attr in bulk_info_str:
                        scenario_generator = \
                            self.__dbInfo.get_variable_generator_function(
                                bulk_attr)(
                                    relation=self.__relation,
                                    base_predicate=bulk_info_str[bulk_attr]
                                )
                        
                        bulk_samples[bulk_attr] = scenario_generator.generate_scenarios(
                            seed=Hyperparameters.INIT_SEED,
                            no_of_scenarios=num_samples
                        )
                        
                        if len(bulk_samples[bulk_attr]) != len(tid_dict[bulk_attr]):
                            print('Something wrong, len(bulk_samples[bulk_attr])=',
                                  len(bulk_samples[bulk_attr]), 'len(tid_dict[bulk_attr])=',
                                  len(tid_dict[bulk_attr]))
                
                counter +=1
                if counter == bulk_sample_size:
                    counter = 0
                
                tid = representatives[(pid, attr)]
                samples = bulk_samples[attr][indices[attr][tid]]
                if len(samples) != num_samples:
                    print('Something wrong, expected', num_samples, 'samples, but got',
                          len(samples), 'samples')
                '''
                individual_scenario_generator = \
                    self.__dbInfo.get_variable_generator_function(attr)(
                        relation=self.__relation,
                        base_predicate='id='+str(tid)
                    )
                
                samples = individual_scenario_generator.generate_scenarios(
                    seed=Hyperparameters.INIT_SEED,
                    no_of_scenarios=num_samples
                )[0]

                
                min_sample = np.min(samples)
                max_sample = np.max(samples)

                if max_sample - min_sample < 1e-8:
                    samples = [1.0 for _ in range(len(samples))]
                
                else:
                    samples = samples - min_sample
                    samples = samples / (max_sample - min_sample)
                    samples = np.sort(samples)
                
                # Step 1: Generate intervals
                samples_per_interval = int((eps_prime*num_samples)/(6*k))
                intervals = []
                
                current_interval_start = 0
                current_interval_end = 0
                samples_in_current_interval = 0
                
                for _ in range(len(samples)):
                    if samples_in_current_interval >= samples_per_interval:
                        current_interval_end = samples[_]
                        if current_interval_end > current_interval_start + 1e-3:
                            if samples[_] == 1.0:
                                while _ < len(samples) and samples[_] == 1.0:
                                    _ = _ + 1
                                    samples_in_current_interval += 1
                                intervals.append(
                                    (current_interval_start, current_interval_end,
                                    samples_in_current_interval/float(
                                        num_samples*(current_interval_end - \
                                                 current_interval_start))
                                    )
                                )
                                samples_in_current_interval = 0
                                break
                            
                            if len(intervals) > 0:
                                last_start, last_end, last_prob = \
                                    intervals[-1]
                                if current_interval_start < last_end:
                                    intervals[-1] = (last_start, current_interval_end,
                                        (last_prob*(last_end - last_start) + samples_in_current_interval)/\
                                            float(num_samples*(current_interval_end - last_start)))
                                else:
                                    intervals.append(
                                       (current_interval_start, current_interval_end,
                                        samples_in_current_interval/float(
                                            num_samples*(current_interval_end - \
                                            current_interval_start))))    
                            else:
                                intervals.append(
                                    (current_interval_start, current_interval_end,
                                    samples_in_current_interval/float(
                                        num_samples*(current_interval_end - \
                                        current_interval_start))
                                    )
                                )
                            
                            current_interval_start = samples[_]
                            samples_in_current_interval = 0
                    samples_in_current_interval += 1
                
                if samples_in_current_interval >= 1:
                    current_interval_end = 1.0
                    if len(intervals) > 0:
                        last_start, last_end, last_prob = \
                            intervals[-1]
                        if current_interval_start < last_end:
                            intervals[-1] = (last_start, current_interval_end,
                                (last_prob*(last_end - last_start) + samples_in_current_interval)/\
                                    float(num_samples*(current_interval_end - last_start)))
                        else:
                            intervals.append(
                                (current_interval_start, current_interval_end,
                                samples_in_current_interval/float(
                                    num_samples*(current_interval_end - \
                                        current_interval_start))))    
                    else:
                        intervals.append(
                            (current_interval_start, current_interval_end,
                            samples_in_current_interval/float(
                            num_samples*(current_interval_end - \
                                current_interval_start))))
                
                # Step 3: Initialize P_0 and F_0 of the algorithm
                prev_p = intervals
                prev_f = []

                # Step 4: Iterate till histogram is produced
                new_p = []
                new_f = []
                num_iter = int(np.ceil(np.log2(1/eps_prime)))
                for _ in range(num_iter):
                    # (a) Initialize p_t and f_t
                    new_p = []
                    new_f = prev_f
                    histogram_changed = False 

                    # (b) Add to f_t
                    num_intervals = len(prev_p)
                    for i in range(num_intervals-1):
                        if prev_p[i] not in prev_f:
                            if prev_p[i+1] not in prev_f:
                                i_start, i_end, i_prob = \
                                    prev_p[i]
                                j_start, j_end, j_prob = \
                                    prev_p[i+1]
                                i_interval = i_end - i_start
                                j_interval = j_end - j_start

                                if i_interval < 1e-8:
                                    continue
                                if j_interval < 1e-8:
                                    continue

                                new_prob = i_interval * i_prob
                                new_prob += j_interval * j_prob
                                new_prob /= (i_interval + \
                                             j_interval)

                                i_diff = new_prob - i_prob
                                j_diff = new_prob - j_prob

                                if i_diff < 0:
                                    i_diff *= -1
                                if j_diff < 0:
                                    j_diff *= -1
                                
                                alpha = i_diff * i_interval
                                alpha += j_diff * j_interval

                                if alpha > (eps_prime/(2*k)):
                                    new_f.append(prev_p[i])
                                    new_f.append(prev_p[i+1])
                                else:
                                    histogram_changed = True
                    
                    # (c) Add to p_t
                    i = 0
                    while i < num_intervals:
                        # Case 1:
                        if i < num_intervals - 1:
                            a, b1, prob_1 = prev_p[i]
                            b2, c, prob_2 = prev_p[i+1]
                            if prev_p[i] not in new_f:
                                if prev_p[i+1] not in new_f:
                                    new_p.append((a, c,\
                                                  ((b1-a)*prob_1 +\
                                                  (c-b2)*prob_2)/(c-a)))
                                    i+=2
                                else:
                                    # Case 3:
                                    new_p.append(prev_p[i])
                                    i+=2
                            else:
                                # Case 2:
                                new_p.append(prev_p[i])
                                i+=1
                        else:
                            # Case 4:
                            new_p.append(prev_p[i])
                            i+=1
                    
                    # (d) Merge new_p and new_f
                    for interval in new_f:
                        if interval not in new_p:
                            new_p.append(interval)
                    
                    new_p.sort()
                    prev_f = new_f
                    prev_p = new_p
                    if not histogram_changed:
                        break

                cdf = []
                prob_widths = []
                bin_widths = []
                bin_starts = []

                cumulative_probability = 0

                for interval in new_p:
                    start, end, prob_height = interval
                    
                    prob = prob_height * (end - start)

                    denormalized_start = start*(
                        max_sample - min_sample) + min_sample
                    denormalized_end = end*(
                        max_sample - min_sample) + min_sample

                    prob_widths.append(prob)
                    cdf.append(cumulative_probability)
                    bin_starts.append(denormalized_start)
                    bin_widths.append(denormalized_end -\
                                      denormalized_start)
                    cumulative_probability += prob

                self.__hist_bins[pid, attr] = []
                for _ in range(len(prob_widths)):
                    if is_first:
                        is_first = False
                    else:
                        insert_sql += ', '
                    # ' (partition_id, attribute, bar_start, bar_width,'\
                    # + ' start_cdf, prob_width) VALUES '
                    insert_sql += '(' + str(pid) + ', ' + "'" +\
                        attr + "', " + str(round(bin_starts[_], 8)) +\
                        ', ' + str(round(bin_widths[_], 8)) + ', ' +\
                        str(round(cdf[_], 8)) + ', ' +\
                        str(round(prob_widths[_], 8)) + ')'

                    self.__hist_bins[pid, attr].append(
                        (bin_starts[_], bin_widths[_], cdf[_], prob_widths[_]))


        PgConnection.Execute(delete_table_sql)
        PgConnection.Execute(create_table_sql)
        PgConnection.Execute(insert_sql)
        PgConnection.Commit()
                
                   


    def __get_stochastic_representative(
        self, partition_id: int, attr: str
    ):
        partition_scenarios = []
        for id in self.__ids_in_partition[partition_id]:
            partition_scenarios.append(self.__scenarios[attr][id])
        scenario_maxes = np.max(partition_scenarios, axis=0)
        scenario_mins = np.min(partition_scenarios, axis=0)
        best_tuple_vectorized = self.__ids_in_partition[partition_id][
            np.argmin(np.sum(np.maximum(
                np.subtract(scenario_maxes, partition_scenarios),
                np.subtract(partition_scenarios, scenario_mins)),
                axis=1))]
        return best_tuple_vectorized


    def __get_deterministic_representative(
        self, partition_id: int, attr: str
    ):
        min_value = None
        max_value = None

        for tuple in self.__ids_in_partition[partition_id]:
            value = self.__values[attr][tuple]
            if max_value is None:
                max_value = value
                min_value = value
            if value > max_value:
                max_value = value
            if value < min_value:
                min_value = value

        best_tuple = None
        best_tuple_distance = None

        for tuple in self.__ids_in_partition[partition_id]:
            value = self.__values[attr][tuple]
            
            dist_from_max = max_value - value
            dist_from_min = value - min_value

            tuple_distance = dist_from_min
            if dist_from_max > dist_from_min:
                tuple_distance = dist_from_max
        
            if best_tuple is None:
                best_tuple = tuple
                best_tuple_distance = tuple_distance
            if best_tuple_distance > tuple_distance:
                best_tuple = tuple
                best_tuple_distance = tuple_distance
        
        return best_tuple            


    def get_partition_representatives(self):
        representatives = dict()
        for p_no in range(self.__no_of_partitions):
            for attr in \
                self.__dbInfo.get_stochastic_attributes(): 
                representatives[(p_no, attr)] = \
                    self.__get_stochastic_representative(
                        p_no, attr
                    )
            for attr in \
                self.__dbInfo.get_deterministic_attributes():
                representatives[(p_no, attr)] = \
                    self.__get_deterministic_representative(
                        p_no, attr
                    )
        
        self.__representative_relation = \
            Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX + \
            self.__relation
        
        drop_table_sql = 'DROP TABLE IF EXISTS ' + \
            self.__representative_relation
        PgConnection.Execute(drop_table_sql)

        create_table_sql = \
            'CREATE TABLE ' + self.__representative_relation\
            + ' (partition_id int not null,'\
            + 'attribute varchar(25) not null,'\
            + 'representative_tuple_id int not null,'\
            + 'PRIMARY KEY(partition_id, attribute));'
        PgConnection.Execute(create_table_sql)

        insert_table_sql = \
            'INSERT INTO ' + self.__representative_relation\
            + " (partition_id, attribute, representative_tuple_id)"\
            + ' VALUES '
        
        is_first = True
        for pid, attr in representatives:
            if is_first:
                is_first = False
            else:
                insert_table_sql += ','
            insert_table_sql += ' (' + str(pid) + \
                ", '" + attr + "', " +\
                str(representatives[(pid, attr)]) + ')'
        
        insert_table_sql += ';'
        PgConnection.Execute(insert_table_sql)
        PgConnection.Commit()
        return representatives

    
    def __partition(self, ids: list[int],
                  depth = 1):
        if len(ids) == 1:
            self.__ids_in_partition.append([])
            self.__ids_in_partition[-1].append(ids[0])
            self.__partition_no[ids[0]] = self.__no_of_partitions
            self.__no_of_partitions += 1
            self.__tuples_whose_partitions_are_found += 1
            return
        
        attributes = []

        for det_attr in self.__dbInfo.get_deterministic_attributes():
            attributes.append(det_attr)
        
        for stoch_attr in self.__dbInfo.get_stochastic_attributes():
            attributes.append(stoch_attr)
        
        farthest_pivot = None
        current_highest_ratio = -1.0
        attribute_with_highest_ratio = None

        for attribute in attributes:
            farthest_distance, pivot = \
                self.get_ids_with_increasing_distances_random_pivot(
                    attribute, ids
                )[-1]
            if farthest_distance / self.__diameter_thresholds[
                attribute] > current_highest_ratio:
                farthest_pivot = pivot
                attribute_with_highest_ratio = attribute
                current_highest_ratio = farthest_distance / \
                    self.__diameter_thresholds[attribute]

        if len(ids) > self.__size_threshold:
            temp_ids = []
            distances_and_ids_from_farthest_pivot = \
                self.get_ids_with_increasing_distances_with_pivot(
                    attribute_with_highest_ratio,
                    ids, farthest_pivot
                )
            for _, id in distances_and_ids_from_farthest_pivot:
                temp_ids.append(id)
                if len(temp_ids) == self.__size_threshold:
                    self.__partition(temp_ids, depth+1)
                    temp_ids = []
            if len(temp_ids) > 0:
                self.__partition(temp_ids, depth+1)
            return
    
        if current_highest_ratio > 1.0:
            
            temp_ids = []
            multiple = 1
            distances_and_ids_from_farthest_pivot = \
                self.get_ids_with_increasing_distances_with_pivot(
                    attribute_with_highest_ratio,
                    ids, farthest_pivot
                )
            for distance, id in distances_and_ids_from_farthest_pivot:
                if distance > multiple*self.__diameter_thresholds[attribute_with_highest_ratio]:
                    self.__partition(temp_ids, depth+1)
                    while distance > multiple*self.__diameter_thresholds[
                        attribute_with_highest_ratio]:
                        multiple += 1
                    temp_ids = []
                temp_ids.append(id)
            
            if len(temp_ids) > 0:
                self.__partition(temp_ids, depth+1)
            return

        self.__ids_in_partition.append([])
        for id in ids:
            self.__partition_no[id] = self.__no_of_partitions
            self.__ids_in_partition[-1].append(id)
        self.__no_of_partitions += 1
        self.__tuples_whose_partitions_are_found += len(ids)
        #print(self.__no_of_partitions, ' partitions formed for',
        #      self.__tuples_whose_partitions_are_found, 'tuples')

    
    def __form_partition_table(self):
        relation = Relation_Prefixes.PARTITION_RELATION_PREFIX +\
            self.__relation
        drop_index_sql = 'DROP INDEX IF EXISTS INDEX_' +\
            relation + '_tuple_index' 
        drop_table_sql = 'DROP TABLE IF EXISTS ' + relation 
        create_table_sql = 'CREATE TABLE ' + relation\
            + ' (partition_id int not null,'\
            + 'tuple_id int not null,'\
            + 'PRIMARY KEY(tuple_id));'
        
        PgConnection.Execute(drop_index_sql)
        PgConnection.Execute(drop_table_sql)
        PgConnection.Execute(create_table_sql)

        insert_sql = 'INSERT INTO ' + relation
        insert_sql += '(partition_id, tuple_id) VALUES'
        is_first = True
        for id in self.__partition_no:
            if is_first:
                is_first = False
            else:
                insert_sql += ','
            insert_sql += ' (' + str(self.__partition_no[id]) + ', ' + \
                str(id) + ')'
        insert_sql += ';'
        PgConnection.Execute(insert_sql)
        
        '''
        index_sql = 'CREATE INDEX  IF NOT EXISTS ' +\
            'INDEX_' + relation + '_tuple_index' +\
            ' ON ' + relation + '(tuple_id);'        
        PgConnection.Execute(index_sql)
        '''
        PgConnection.Commit()
        print('Formed partitioned tables')

    
    def get_no_of_partitions(self):
        return self.__no_of_partitions