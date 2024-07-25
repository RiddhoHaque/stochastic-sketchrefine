import numpy as np
from Utils.Databases import Databases
from DbInfo.DbInfo import DbInfo
from DbInfo.PortfolioInfo import PortfolioInfo
from DbInfo.TpchInfo import TpchInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OfflinePreprocessing.MeanAbsoluteDistance import MeanAbsoluteDistance
from ScenarioGenerator.PorfolioScenarioGenerator.GainScenarioGenerator import GainScenarioGenerator
from ScenarioGenerator.TpchScenarioGenerators.PriceScenarioGenerator import PriceScenarioGenerator
from ScenarioGenerator.TpchScenarioGenerators.QuantityScenarioGenerator import QuantityScenarioGenerator
from ValueGenerator.ValueGenerator import ValueGenerator


class PivotScan:

    @staticmethod
    def __get_values(relation: str,
                   interval_start: int,
                   interval_end: int,
                   attribute: str):
        return ValueGenerator(
            relation=relation,
            base_predicate='id >= ' + \
                str(interval_start) + \
                ' and id <= ' + \
                str(interval_end),
            attribute=attribute
        ).get_values()
    
    @staticmethod
    def __get_scenarios(relation: str,
                      interval_start: int,
                      interval_end: int,
                      attribute: str,
                      init_seed: int,
                      dbinfo: DbInfo):
        vg_function = dbinfo.get_variable_generator_function(attribute)
        return vg_function(
                relation=relation,
                base_predicate='id >= ' + str(interval_start) + \
                ' and id <= ' + str(interval_end)
            ).generate_scenarios(
                seed=init_seed,
                no_of_scenarios = Hyperparameters.MAD_NO_OF_SAMPLES)
    
    @staticmethod
    def __add_to_combined_values(
        relation: str,
        interval_start: int,
        interval_end: int,
        attribute: str,
        combined_values: list[float]
    ):
        values = PivotScan.__get_values(
            relation, interval_start,
            interval_end, attribute
        )
        for value in values:
            combined_values.append(value[0])
    
    @staticmethod
    def __add_to_combined_scenarios(
        relation: str,
        interval_start: int,
        interval_end: int,
        attribute: str,
        init_seed: int,
        dbinfo: DbInfo,
        combined_scenarios: list[list[float]]
    ):
        scenarios = PivotScan.__get_scenarios(
            relation,
            interval_start,
            interval_end,
            attribute,
            init_seed,
            dbinfo
        )
        for scenario in scenarios:
            combined_scenarios.append(
                scenario)
    
    @staticmethod
    def __process_combined_interval(
        relation: str,
        interval_start: int,
        interval_end: int,
        attribute: str,
        init_seed: int,
        dbinfo: DbInfo,
        combined_values: list[float],
        combined_scenarios: list[list[float]]
    ):
        if dbinfo.is_deterministic_attribute(
            attribute=attribute
        ):
            PivotScan.__add_to_combined_values(
                relation, interval_start,
                interval_end, attribute,
                combined_values
            )
        else:
            PivotScan.__add_to_combined_scenarios(
                relation, interval_start,
                interval_end, attribute, init_seed,
                dbinfo, combined_scenarios
            )


    @staticmethod
    def get_ids_with_increasing_distances(
        ids: list[int], pivots: list[int],
        attribute: str, relation: str,
        dbinfo: DbInfo, init_seed: int,
        get_distances_from_farthest_tuple = False,
        diameter_threshold = None) -> list[(float, int)]:
        
        ids.sort()

        first_id = ids[0]
        current_interval_start = first_id
        current_interval_end = first_id
        combined_values = []
        combined_scenarios = []
        
        for id in ids:
            if id == current_interval_end:
                continue
            elif id == current_interval_end + 1:
                current_interval_end = id
            elif id > current_interval_end + 1:
                PivotScan.__process_combined_interval(
                    relation, current_interval_start,
                    current_interval_end, attribute,
                    init_seed, dbinfo, combined_values,
                    combined_scenarios)
                current_interval_start = id
                current_interval_end = id
        
        PivotScan.__process_combined_interval(
            relation, current_interval_start,
            current_interval_end, attribute,
            init_seed, dbinfo, combined_values,
            combined_scenarios)
        
        id_distance_pairs = []
        if dbinfo.is_deterministic_attribute(attribute):
            counter = 0
            for pivot in pivots:
                for idx in range(len(ids)):
                    id_distance_pairs.append(
                        (np.abs(combined_values[idx] - \
                                combined_values[pivot]),
                        ids[idx])
                    )
                counter += 1
            
        else:
            counter = 0
            for pivot in pivots:
                for idx in range(len(ids)):
                    id_distance_pairs.append((
                        np.average(np.abs(
                            np.subtract(
                                combined_scenarios[idx],
                                combined_scenarios[pivot]))),
                        ids[idx]))
                counter += 1
        id_distance_pairs.sort()
        
        repivot = False
        if get_distances_from_farthest_tuple:
            if diameter_threshold is not None:
                farthest_distance, _ = id_distance_pairs[-1]
                if farthest_distance > diameter_threshold:
                    repivot = True
            else:
                repivot = True
        if repivot:
            pivots = [len(id_distance_pairs)-1]
            id_distance_pairs = []
            if dbinfo.is_deterministic_attribute(attribute):
                counter = 0
                for pivot in pivots:
                    for idx in range(len(ids)):
                        id_distance_pairs.append(
                            (np.abs(combined_values[idx] - \
                                    combined_values[pivot]),
                            ids[idx])
                        )
                    counter += 1
            else:
                counter = 0
                for pivot in pivots:
                    for idx in range(len(ids)):
                        id_distance_pairs.append((
                            np.average(np.abs(
                                np.subtract(
                                    combined_scenarios[idx],
                                    combined_scenarios[pivot]))),
                            ids[idx]))
                    counter += 1
            id_distance_pairs.sort()
        return id_distance_pairs
