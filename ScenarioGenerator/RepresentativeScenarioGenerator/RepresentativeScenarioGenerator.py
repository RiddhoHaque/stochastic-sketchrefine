import time
import numpy as np
from numpy.random import SFC64, SeedSequence, Generator
from scipy.stats import norm

from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGeneratorWithoutCorrelation import RepresentativeScenarioGeneratorWithoutCorrelation
from SeedManager.SeedManager import SeedManager
from Utils.Relation_Prefixes import Relation_Prefixes


class RepresentativeScenarioGenerator(ScenarioGenerator):
    __cached_histograms = dict()

    def __init__(self, 
                 relation: str,
                 attr: str,
                 dbInfo: DbInfo,
                 base_predicate = '',
                 duplicate_vector = [],
                 correlation_coeff = [0.01]) -> None:
        self.__relation = relation
        self.__attribute = attr
        self.__dbInfo = dbInfo
        self.__base_predicate = base_predicate
        self.__duplicate_vector = duplicate_vector
        self.__correlation_coeff = correlation_coeff

    
    def __get_partition_ids(self):
        representative_relation = \
            Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX +\
                self.__relation
        sql = 'SELECT DISTINCT partition_id FROM ' +\
            representative_relation + ' WHERE attribute=' +\
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

    def __generate_values(self, pid, n_values):

        cache_key = (self.__relation, self.__attribute, pid)
        if cache_key in RepresentativeScenarioGenerator.__cached_histograms:
            cdf, edges = RepresentativeScenarioGenerator.__cached_histograms[cache_key]

        else:
            num_scenarios = Hyperparameters.NO_OF_HIST_BINS*10

            scenarios = RepresentativeScenarioGeneratorWithoutCorrelation(
                relation=self.__relation,
                attr=self.__attribute,
                scenario_generator=self.__dbInfo.get_variable_generator_function(
                    self.__attribute),
                duplicates=[1],
                base_predicate='partition_id=' + str(pid)
            ).generate_scenarios(
                seed=Hyperparameters.INIT_SEED,
                no_of_scenarios=int(num_scenarios),
            )

            scenarios = np.array(scenarios)

            bins = Hyperparameters.NO_OF_HIST_BINS

            edges = np.quantile(scenarios, np.linspace(0,1, bins+1))
            counts, _ = np.histogram(scenarios, bins=edges)

            probs = counts.astype(float)
            probs /= probs.sum()

            cdf = np.cumsum(probs)
            RepresentativeScenarioGenerator.__cached_histograms[cache_key] = cdf, edges

        rng = np.random.default_rng(seed=SeedManager.get_next_seed())

        # uniform values for inverse CDF sampling
        u = rng.random(n_values)

        # find bins
        bin_idx = np.searchsorted(cdf, u, side="right")
        bin_idx = np.clip(bin_idx, 0, len(cdf) - 1)

        cdf_left = np.where(bin_idx == 0, 0.0, cdf[bin_idx - 1])
        cdf_right = cdf[bin_idx]

        left = edges[bin_idx]
        right = edges[bin_idx + 1]
            
        # interpolate within bin
        values = left + (u - cdf_left) * (right - left) / (cdf_right - cdf_left)

        return values

    def generate_scenarios(
        self, seed: int, no_of_scenarios: int, pid=None,
        duplicates_to_use=-1, correlation_to_use=-2
    ) -> list[list[float]]:

        scenarios = []
        rng = Generator(SFC64(SeedSequence(seed)))

        if pid is None:
            pids = self.__get_partition_ids()
        else:
            pids = [pid]

        for idx, pid in enumerate(pids):

            if duplicates_to_use == -1:
                duplicates = self.__duplicate_vector[idx]
            else:
                duplicates = duplicates_to_use

            if correlation_to_use == -2:
                correlation = self.__correlation_coeff[idx]
            else:
                correlation = correlation_to_use

            duplicates += 1

            # Step 1: Generate independent normals
            norta_vec = rng.standard_normal(
                size=(int(duplicates), no_of_scenarios)
            )

            # Step 2: Apply NORTA correlation structure
            lambda_1 = np.sqrt(max(0.0, 1 + (duplicates - 1) * correlation))
            lambda_2 = np.sqrt(max(0.0, 1 - correlation))

            sum_others = np.sum(norta_vec[1:], axis=0)

            original_norta_0 = norta_vec[0].copy()
            norta_vec[0] = norta_vec[0] * lambda_1 - sum_others * lambda_2

            norta_vec[1:] *= lambda_2
            norta_vec[1:] -= original_norta_0 * lambda_1

            # Normalize rows to unit variance so norm.cdf gives correct uniform marginals.
            # Var(row 0)  = lambda_1^2 + (d-1)*lambda_2^2 = d
            # Var(row i)  = lambda_2^2 + lambda_1^2       = 2 + (d-2)*rho
            norta_vec[0] /= np.sqrt(duplicates)
            if duplicates > 1:
                norta_vec[1:] /= np.sqrt(2 + (duplicates - 2) * correlation)

            # Step 3: Convert to uniform
            u = norm.cdf(norta_vec[1:])

            # Step 4: Flatten and sort ranks
            u_flat = u.ravel()
            order = np.argsort(u_flat)

            # Step 5: Generate marginal samples
            values = self.__generate_values(pid, len(u_flat))

            values.sort()

            # Step 6: Assign according to rank order
            result_flat = np.empty_like(u_flat)
            result_flat[order] = values

            result = result_flat.reshape(u.shape)

            scenarios.extend(result)

        return scenarios

    def generate_scenarios_multiple_pids(
        self, seed: int, no_of_scenarios: int,
        pids: list[int],
        duplicates: list[int],
        correlations_list: list[float]):
        scenarios = []
        for _ in range(len(pids)):
            pid = pids[_]
            new_scenarios = self.generate_scenarios(
                seed, no_of_scenarios, pid,
                duplicates[_],
                correlations_list[_]
            )
            for scenario in new_scenarios:
                scenarios.append(scenario)
        return scenarios