from scenario_generation_demo import gen_scenarios
from DbInfo.PortfolioInfo import PortfolioInfo
from DbInfo.TpchInfo import TpchInfo
from Naive.Naive import Naive
from SummarySearch.SummarySearch import SummarySearch
from OfflinePreprocessing.DistPartition import DistPartition
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.PorfolioScenarioGenerator.GainScenarioGenerator import GainScenarioGenerator
from ScenarioGenerator.TpchScenarioGenerators.PriceScenarioGenerator import PriceScenarioGenerator
from OfflinePreprocessing.MonotonicDequeUnitTest import MonotonicDequeUnitTest
from OfflinePreprocessing.OptimalPartitioningUnitTest import OptimalPartitioningUnitTest
from StochasticPackageQuery.Parser.Parser import Parser
from Utils.Stochasticity import Stochasticity
from UnitTestRunner import UnitTestRunner
import warnings
import time
import os
import numpy as np


if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    workload_directory = 'Workloads/PortfolioWorkload'
    for file in os.listdir(workload_directory):
        with open(
            workload_directory + '/' + file, 'r') as f:
            query = Parser().parse(f.readlines())
            summarySearch = SummarySearch(
                query=query, linear_relaxation=False,
                dbInfo=PortfolioInfo, init_no_of_scenarios=100,
                init_no_of_summaries=1,
                no_of_validation_scenarios=1000000,
                approximation_bound=0.05)
            package, objective_value = summarySearch.solve()
            print('Objective Value', objective_value)
            summarySearch.display_package(package)
    UnitTestRunner()