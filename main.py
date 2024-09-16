from scenario_generation_demo import gen_scenarios
from CVaRification.CVaRification import CVaRification
from CVaRification.StaircaseCVaRification import StaircaseCVaRification
from CVaRification.RCLSolve import RCLSolve
from DbInfo.PortfolioInfo import PortfolioInfo
from DbInfo.TpchInfo import TpchInfo
from Naive.Naive import Naive
from SummarySearch.SummarySearch import SummarySearch
from OfflinePreprocessing.DistPartition import DistPartition
from PgConnection.PgConnection import PgConnection
from QueryHardness.HardnessEvaluator import HardnessEvaluator
from ScenarioGenerator.PorfolioScenarioGenerator.GainScenarioGenerator import GainScenarioGenerator
from ScenarioGenerator.TpchScenarioGenerators.PriceScenarioGenerator import PriceScenarioGenerator
from SeedManager.SeedManager import SeedManager
from OfflinePreprocessing.MonotonicDequeUnitTest import MonotonicDequeUnitTest
from OfflinePreprocessing.OptimalPartitioningUnitTest import OptimalPartitioningUnitTest
from StochasticPackageQuery.Parser.Parser import Parser
from Utils.Stochasticity import Stochasticity
from UnitTestRunner import UnitTestRunner
from ValueGenerator.ValueGenerator import ValueGenerator
from Validator.Validator import Validator
import warnings
import time
import os
import numpy as np


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    '''
    partitioner = DistPartition(
        relation='Stock_Investments_Half',
        dbInfo=PortfolioInfo
    )

    id_lists = ValueGenerator(
        relation='Stock_Investments_Half',
        base_predicate='',
        attribute='id'
    ).get_values()
    
    ids = []

    for tuple in id_lists:
        ids.append(tuple[0])
    
    partitioner.partition(ids)
    print('Number of partitions:', partitioner.get_no_of_partitions())
    '''
    iter = 0
    workload_directory = 'Workloads/PortfolioWorkload'
    for file in os.listdir(workload_directory):
        with open(
            workload_directory + '/' + file, 'r') as f:
            query = Parser().parse(f.readlines())
            SeedManager.reinitialize_seed()
            rclSolve = RCLSolve(
                query=query, linear_relaxation=False,
                dbInfo=PortfolioInfo,
                init_no_of_scenarios=100,
                no_of_validation_scenarios=1000000,
                approximation_bound=0.02,
                sampling_tolerance=0.2,
                bisection_threshold=0.01,
            )
            eval = HardnessEvaluator(
                query=query,
                solver=rclSolve,
                dbInfo=PortfolioInfo,
                validator= Validator(
                    query=query, dbInfo=PortfolioInfo,
                    no_of_validation_scenarios=10000,
                ),
                foraging_period=2,
            )

            eval.log_hardness(iter+1, 'Porfolio')
            #rclSolve.solve()
            #rclMetrics = rclSolve.get_metrics()
            '''
            lpRcl = RCLSolve(
                query=query, linear_relaxation=True,
                dbInfo=PortfolioInfo,
                init_no_of_scenarios=100,
                no_of_validation_scenarios=1000000,
                approximation_bound=0.02,
                sampling_tolerance=0.20,
                bisection_threshold=0.1,
            )
            lpRcl.solve()
            lpRclMetrics = lpRcl.get_metrics()
            SeedManager.reinitialize_seed()
            start_time = time.time()
            summarySearch = SummarySearch(
                query=query, linear_relaxation=False,
                dbInfo=PortfolioInfo, init_no_of_scenarios=100,
                init_no_of_summaries=1,
                no_of_validation_scenarios=1000000,
                approximation_bound=0.02)
            package, objective_value = summarySearch.solve()
            summarySearch.display_package(package)
            summarySearchMetrics = summarySearch.get_metrics()
            SeedManager.reinitialize_seed()
            lpSummarySearch = SummarySearch(
                query=query, linear_relaxation=True,
                dbInfo=PortfolioInfo, init_no_of_scenarios=100,
                init_no_of_summaries=1,
                no_of_validation_scenarios=1000000,
                approximation_bound=0.02)
            lpSummarySearch.solve()
            lpSearchMetrics = lpSummarySearch.get_metrics()
            '''
            iter += 1
            #rclMetrics.log()
            #summarySearchMetrics.log()
            #lpRclMetrics.log()
            #lpSearchMetrics.log()