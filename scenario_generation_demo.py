from ScenarioGenerator.TpchScenarioGenerators.PriceScenarioGenerator import PriceScenarioGenerator
from ScenarioGenerator.PorfolioScenarioGenerator.GainScenarioGenerator import GainScenarioGenerator
from UnitTestRunner import UnitTestRunner
import multiprocessing as mp
from multiprocessing import Process
import numpy as np
import time
import warnings

class Args:

    def __init__(self, relation, base_predicate,
                 seed, no_of_scenarios):
        self.__relation = relation
        self.__base_predicate = base_predicate
        self.__seed = seed
        self.__no_of_scenarios = no_of_scenarios
    
    def get_relation(self):
        return self.__relation
    
    def get_base_predicate(self):
        return self.__base_predicate
    
    def get_seed(self):
        return self.__seed
    
    def get_no_of_scenarios(self):
        return self.__no_of_scenarios


def gen_prices(args):
    warnings.filterwarnings('ignore')
    price_generator = PriceScenarioGenerator(
        relation=args.get_relation(),
        base_predicate=args.get_base_predicate()
    )
    price_generator.generate_scenario(
        seed=args.get_seed(),
        no_of_scenarios=args.get_no_of_scenarios()
    )

def gen_gains(args):
    warnings.filterwarnings('ignore')
    gain_generator = GainScenarioGenerator(
        relation=args.get_relation(),
        base_predicate=args.get_base_predicate()
    )
    gain_generator.generate_scenario(
        seed=args.get_seed(),
        no_of_scenarios=args.get_no_of_scenarios()
    )

def gen_scenarios():
    start_time = time.time()
    seed = 1204567 
    scenario_count = 0
    processes = []
    scenarios_per_process = int(np.floor(100/mp.cpu_count()))
    while scenario_count < 100:
        scenario_per_process = scenarios_per_process
        if len(processes) <= (100%mp.cpu_count()):
            scenario_per_process += 1
        scenario_count += scenarios_per_process
        args = Args(relation='Lineitem_20000',
                    base_predicate='',
                    seed = seed,
                    no_of_scenarios=scenarios_per_process)
        seed+=1
        p = Process(target=gen_prices, args=(args,))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print('Time taken to generate 1 million scenarios of price from 20 thousand tuples:',
          round(time.time() - start_time, 2))
    start_time = time.time()
    seed = 1204567 
    scenario_count = 0
    processes = []
    scenarios_per_process = int(np.floor(100/mp.cpu_count()))
    while scenario_count < 100:
        scenario_per_process = scenarios_per_process
        if len(processes) <= (100%mp.cpu_count()):
            scenario_per_process += 1
        scenario_count += scenarios_per_process
        args = Args(relation='Stock_Investments_Volatility_1x',
                    base_predicate='',
                    seed = seed,
                    no_of_scenarios=scenarios_per_process)
        seed+=1
        p = Process(target=gen_gains, args=(args,))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print('Time taken to generate 1 million scenarios of gain from 20 thousand tuples:',
          round(time.time() - start_time, 2))
