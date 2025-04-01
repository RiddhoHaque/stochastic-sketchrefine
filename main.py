from scenario_generation_demo import gen_scenarios
from CVaRification.CVaRification import CVaRification
from CVaRification.StaircaseCVaRification import StaircaseCVaRification
from CVaRification.RCLSolve import RCLSolve
from DbInfo.PortfolioInfo import PortfolioInfo
from DbInfo.TpchInfo import TpchInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from Naive.Naive import Naive
from SummarySearch.SummarySearch import SummarySearch
from OfflinePreprocessing.DistPartition import DistPartition
from PgConnection.PgConnection import PgConnection
from QueryHardness.HardnessEvaluator import HardnessEvaluator
from QueryHardness.RCLSolveBasedHardness import RCLSolveBasedHardness
from ScenarioGenerator.PorfolioScenarioGenerator.GainScenarioGenerator import GainScenarioGenerator
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGenerator import RepresentativeScenarioGenerator
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGeneratorWithoutCorrelation import RepresentativeScenarioGeneratorWithoutCorrelation
from ScenarioGenerator.TpchScenarioGenerators.PriceScenarioGenerator import PriceScenarioGenerator
from SeedManager.SeedManager import SeedManager
from SketchRefine.Sketch import Sketch
from SketchRefine.SketchRefine import SketchRefine
from OfflinePreprocessing.MonotonicDequeUnitTest import MonotonicDequeUnitTest
from OfflinePreprocessing.OptimalPartitioningUnitTest import OptimalPartitioningUnitTest
from StochasticPackageQuery.Parser.Parser import Parser
from Utils.Stochasticity import Stochasticity
from Utils.Relation_Prefixes import Relation_Prefixes
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
    st1 = time.time()
    scenarios = representative_sg = RepresentativeScenarioGenerator(
        relation='Stock_Investments_Half',
        base_predicate='partition_id=1',
        duplicate_vector=[20],
        correlation_coeff=0.5
    ).generate_scenarios(seed=203545, no_of_scenarios=200)
    print('Took', time.time() - st1, 'secs')
    st2 = time.time()
    scenarios = representative_sg = RepresentativeScenarioGenerator(
        relation='Stock_Investments_Half',
        base_predicate='partition_id=1',
        duplicate_vector=[20],
        correlation_coeff=0.5
    ).generate_scenarios(seed=203545, no_of_scenarios=200, pid=1)
    print('Took', time.time() - st2, 'secs')
    print(scenarios[15][156])
    '''

    relation = 'Lineitem_6000000'
    count_query = 'SELECT COUNT(*) from ' + relation
    PgConnection.Execute(count_query)
    no_of_tuples = PgConnection.Fetch()[0][0]
    
    workload_directory = 'Workloads/PortfolioWorkload'
    for file in os.listdir(workload_directory):
        with open(workload_directory + '/' + file, 'r') as f:
            query = Parser().parse(f.readlines())
            validation_scenarios_list = [1000000]
            result_dict = dict()
            for no_of_validation_scenarios in validation_scenarios_list:
                variances = []
                for _ in range(30):
                    tuples = [_ for _ in range(no_of_tuples)]
                    np.random.shuffle(tuples)
                    tuples = tuples[0:5]
                    package_dict = dict()
                    for tuple_id in tuples:
                        package_dict[tuple_id] = 1
                    objective_values = []

                    for _ in range(30):
                        validator = Validator(
                            query=query, dbInfo=PortfolioInfo,
                            no_of_validation_scenarios=no_of_validation_scenarios)
                        objective_value = validator.get_validation_objective_value(package_dict)
                        objective_values.append(objective_value)
                    
                    variances.append(np.var(objective_values))
                result_dict[no_of_validation_scenarios] = np.mean(variances)
                print('Mean Variance:', np.mean(variances))
            print(result_dict)
        break


    '''
    sg1 = RepresentativeScenarioGeneratorWithoutCorrelation(
           relation='Stock_Investments_Half',
           attr='gain', base_predicate='partition_id=49085',
           duplicates=[5], scenario_generator=GainScenarioGenerator
    )

    #sg1 = RepresentativeScenarioGenerator(
    #    relation='Stock_Investments_Half',
    #    attr='gain', base_predicate='partition_id=49085',
    #    duplicate_vector=[5], correlation_coeff=[-0.1]
    #)
    
    scenarios = sg1.generate_scenarios(
        seed=Hyperparameters.INIT_SEED,
        no_of_scenarios=100)
    
    scenarios_1 = scenarios[0]
    scenarios_2 = scenarios[1]
    scenarios_3 = scenarios[2]
    scenarios_4 = scenarios[3]
    scenarios_5 = scenarios[4]
    
    sql = 'SELECT representative_tuple_id FROM ' +\
        Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX +\
        "Stock_Investments_Half WHERE partition_id=49085 AND attribute='gain'"
    PgConnection.Execute(sql)
    tid = PgConnection.Fetch()[0][0]
    print('tid:', tid)

    sg2 = GainScenarioGenerator(
        relation='Stock_Investments_Half',
        base_predicate='id='+str(tid)
    )

    scenarios_6 = sg2.generate_scenarios(
        seed=Hyperparameters.INIT_SEED, no_of_scenarios=100)[0]
    
    print('Avg. of scenarios 1:', np.average(scenarios_1))
    print('Avg. of scenarios 2:', np.average(scenarios_2))
    print('Avg. of scenarios 3:', np.average(scenarios_3))
    print('Avg. of scenarios 4:', np.average(scenarios_4))
    print('Avg. of scenarios 5:', np.average(scenarios_5))
    print('Avg. of scenarios 6:', np.average(scenarios_6))
    
    until = 10
    while until <= 10:
        print('CVaR in lower', until/100.0, 'tail')
        print('Avg. of Scenarios 1:', np.average(np.sort(scenarios_1)[:until]))
        print('Avg, of Scenarios 2:', np.average(np.sort(scenarios_2)[:until]))
        print('Avg, of Scenarios 3:', np.average(np.sort(scenarios_3)[:until]))
        print('Avg, of Scenarios 4:', np.average(np.sort(scenarios_4)[:until]))
        print('Avg, of Scenarios 5:', np.average(np.sort(scenarios_5)[:until]))
        print('Avg. of scenarios 6:', np.average(np.sort(scenarios_6)[:until]))
        until += 1000
    '''
    '''
    partitioner = DistPartition(
        relation='stock_investments_half',
        dbInfo=PortfolioInfo
    )
    partitioner.partition_relation()
    partitioner.get_metrics().log_performance()
    
    
    partitioner = DistPartition(
        relation='lineitem_6000000',
        dbInfo=TpchInfo
    )
    partitioner.partition_relation()
    partitioner.get_metrics().log_performance()
    '''
    '''
    iter = 0
    workload_directory = 'Workloads/PortfolioWorkload'
    for file in os.listdir(workload_directory):
        with open(
            workload_directory + '/' + file, 'r') as f:
            query = Parser().parse(f.readlines())
            relations = ['stock_investments_half']
            
            for relation in relations:
                query.set_relation(relation)
                SeedManager.reinitialize_seed()
                package_dict, objective_value =\
                SketchRefine(query, PortfolioInfo).\
                    solve()
                print('Sketch package:', package_dict,
                    'Objective value:', objective_value)
            
    '''
    '''
                hardness_evaluator =\
                    RCLSolveBasedHardness(
                        query=query, linear_relaxation=False,
                        dbInfo=PortfolioInfo,
                        init_no_of_scenarios=100,
                        no_of_validation_scenarios=1000000,
                        approximation_bound=0.05,
                        sampling_tolerance=1.00,
                        bisection_threshold=0.1
                    )
                hardness_evaluator.solve()
                print('Hardness = ',
                      hardness_evaluator.get_model_probability())
                #rclSolve.solve()
                #rclMetrics = rclSolve.get_metrics()
    '''
    '''
                rcl = RCLSolve(
                    query=query, linear_relaxation=False,
                    dbInfo=TpchInfo,
                    init_no_of_scenarios=100,
                    no_of_validation_scenarios=1000000,
                    approximation_bound=0.05,
                    sampling_tolerance=0.20,
                    bisection_threshold=0.1,
                )
                rcl.solve()
                rclMetrics = rcl.get_metrics()
                print(file)
                rclMetrics.log()
                
    '''
    '''
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
    '''
    '''
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
    #iter += 1
    #rclMetrics.log()
    #summarySearchMetrics.log()
    #lpRclMetrics.log()
    #lpSearchMetrics.log()
