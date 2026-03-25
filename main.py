from CVaRification.RCLSolve import RCLSolve
from DbInfo.DbInfo import DbInfo
from DbInfo.PortfolioInfo import PortfolioInfo
from DbInfo.TpchInfo import TpchInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from Naive.Naive import Naive
from SummarySearch.SummarySearch import SummarySearch
from SketchRefine.SketchRefine import SketchRefine
from StochasticPackageQuery.Parser.Parser import Parser
from StochasticPackageQuery.Query import Query
import multiprocessing
import warnings
import os
import numpy as np
import sys

TIMEOUT = 120*60

def run_naive_lp(
    query: Query, dbInfo: DbInfo,
    result_queue: multiprocessing.Queue
):
    naive_solver = Naive(
        query=query, linear_relaxation=True, dbInfo=dbInfo,
        init_no_of_scenarios=Hyperparameters.INIT_NO_OF_SCENARIOS,
        no_of_validation_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
        approximation_bound=Hyperparameters.APPROXIMATION_BOUND
    )
    naive_package, upper_bound = naive_solver.solve()
    print('Naive package:', naive_package)
    print('Naive upper bound:', upper_bound)
    result_queue.put(upper_bound)

def run_summarysearch_lp(
    query: Query, dbInfo: DbInfo,
    result_queue: multiprocessing.Queue
):
    ssearch_solver = SummarySearch(
        query=query, linear_relaxation=True, dbInfo=dbInfo,
        init_no_of_scenarios=Hyperparameters.INIT_NO_OF_SCENARIOS,
        init_no_of_summaries=Hyperparameters.INIT_NO_OF_SUMMARIES,
        no_of_validation_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
        approximation_bound=Hyperparameters.APPROXIMATION_BOUND,
        max_opt_scenarios=Hyperparameters.MAX_OPT_SCENARIOS_IN_PRACTICE
    )
    package, upper_bound = ssearch_solver.solve()
    print('SSearch LP Package:', package)
    print('SSearch Upper Bound:', upper_bound)
    result_queue.put(upper_bound)

def run_rclsolve_lp(
    query: Query, dbInfo: DbInfo,
    result_queue: multiprocessing.Queue
):
    rclsolver = RCLSolve(
        query=query, linear_relaxation=True, dbInfo=dbInfo,
        init_no_of_scenarios=Hyperparameters.INIT_NO_OF_SCENARIOS,
        no_of_validation_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
        approximation_bound=Hyperparameters.APPROXIMATION_BOUND,
        sampling_tolerance=Hyperparameters.SAMPLING_TOLERANCE,
        bisection_threshold=Hyperparameters.BISECTION_THRESHOLD,
        max_opt_scenarios=Hyperparameters.MAX_OPT_SCENARIOS_IN_PRACTICE
    )
    package, upper_bound = rclsolver.solve()
    print('RCLSolve LP Package:', package)
    print('RCLSolve Upper Bound:', upper_bound)
    result_queue.put(upper_bound)


def get_lp_upper_bound(query: Query, dbInfo: DbInfo):
    result_queue = multiprocessing.Queue()

    naive_lp_process = multiprocessing.Process(
        target=run_naive_lp,
        args=(query, dbInfo, result_queue)
    )
    naive_lp_process.start()
    naive_lp_process.join(timeout=TIMEOUT)

    if naive_lp_process.is_alive():
        naive_lp_process.terminate()
        naive_lp_process.join()
        summary_search_process = multiprocessing.Process(
            target=run_summarysearch_lp,
            args=(query, dbInfo, result_queue)
        )
        summary_search_process.start()
        summary_search_process.join(timeout=TIMEOUT)

        if summary_search_process.is_alive():
            summary_search_process.terminate()
            summary_search_process.join()
            rclsolve_lp_process = multiprocessing.Process(
                target=run_rclsolve_lp,
                args=(query, dbInfo, result_queue)
            )
            rclsolve_lp_process.start()
            rclsolve_lp_process.join(timeout=TIMEOUT)
            if rclsolve_lp_process.is_alive():
                rclsolve_lp_process.terminate()
                rclsolve_lp_process.join()
                return 0.0
            return result_queue.get()
        else:
            return result_queue.get()
    else:
        return result_queue.get()

def run_summarysearch(query, dbInfo, result_queue):
    solver = SummarySearch(
        query=query, linear_relaxation=False, dbInfo=dbInfo,
        init_no_of_scenarios=Hyperparameters.INIT_NO_OF_SCENARIOS,
        init_no_of_summaries=Hyperparameters.INIT_NO_OF_SUMMARIES,
        no_of_validation_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
        approximation_bound=Hyperparameters.APPROXIMATION_BOUND,
        max_opt_scenarios=Hyperparameters.MAX_OPT_SCENARIOS_IN_PRACTICE)
    solver.solve()
    metric = solver.get_metrics()
    result_queue.put((metric.get_runtime(), metric.get_objective_value()))


def run_rclsolve(query, dbInfo, result_queue):
    solver = RCLSolve(
        query=query, linear_relaxation=False, dbInfo=dbInfo,
        init_no_of_scenarios=Hyperparameters.INIT_NO_OF_SCENARIOS,
        no_of_validation_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
        approximation_bound=Hyperparameters.APPROXIMATION_BOUND,
        sampling_tolerance=Hyperparameters.SAMPLING_TOLERANCE,
        bisection_threshold=Hyperparameters.BISECTION_THRESHOLD,
        max_opt_scenarios=Hyperparameters.MAX_OPT_SCENARIOS_IN_PRACTICE)
    solver.solve()
    metric = solver.get_metrics()
    result_queue.put((metric.get_runtime(), metric.get_objective_value()))


def run_sskr(query, dbInfo, result_queue):
    solver = SketchRefine(
        query=query, dbInfo=dbInfo, is_lp_relaxation=False)
    solver.solve()
    metric = solver.get_metrics()
    result_queue.put((metric.get_runtime(), metric.get_objective_value()))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    workload_directory = ''

    variance_relations = []
    inc_tuples_relations = []
    dbInfo = None

    if sys.argv[2] == 'tpch':
        workload_directory = 'Workloads/TpchWorkload'
        dbInfo = TpchInfo
        variance_relations = [
            'Lineitem_Variance_1x',
            'Lineitem_Variance_2x',
            'Lineitem_Variance_5x',
            'Lineitem_Variance_8x',
            'Lineitem_Variance_10x',
            'Lineitem_Variance_13x',
            'Lineitem_Variance_17x',
            'Lineitem_Variance_20x',
        ]
        inc_tuples_relations = [
            'Lineitem_20000',
            'Lineitem_60000',
            'Lineitem_120000',
            'Lineitem_300000',
            'Lineitem_450000',
            'Lineitem_600000',
            'Lineitem_1200000',
            'Lineitem_3000000',
            'Lineitem_4500000',
            'Lineitem_6000000',
        ]

    elif sys.argv[2] == 'portfolio':
        workload_directory = 'Workloads/PortfolioWorkload'
        dbInfo = PortfolioInfo
        variance_relations = [
            'Stock_Investments_Volatility_1x',
            'Stock_Investments_Volatility_2x',
            'Stock_Investments_Volatility_5x',
            'Stock_Investments_Volatility_8x',
            'Stock_Investments_Volatility_10x',
            'Stock_Investments_Volatility_13x',
            'Stock_Investments_Volatility_17x',
            'Stock_Investments_Volatility_20x',
        ]
        inc_tuples_relations = [
            'Stock_Investments_90',
            'Stock_Investments_45',
            'Stock_Investments_30',
            'Stock_Investments_15',
            'Stock_Investments_9',
            'Stock_Investments_3',
            'Stock_Investments_1',
            'Stock_Investments_half',
        ]


    query_no = int(sys.argv[4])
    iterations = int(sys.argv[5])

    file_no = 0
    for file in sorted(os.listdir(workload_directory)):
        file_no += 1
        if file_no != query_no:
            continue
        with open(workload_directory + '/' + file, 'r') as f:
            query = Parser().parse(f.readlines())
            print('Executing', file)
            if sys.argv[1] == 'inc_variances':
                for relation in variance_relations:
                    if sys.argv[3] == 'sskr':
                        print('sskr is not supported for inc_variances')
                        continue
                    runtimes = []
                    errors = []
                    timed_out = False
                    if sys.argv[3] == 'rclsolve':
                        print('Getting LP Upper Bound.')
                        query.set_relation(relation)
                        lp_bound = get_lp_upper_bound(query, dbInfo)
                        print('LP Upper Bound =', lp_bound)
                    for _ in range(iterations):
                        query.set_relation(relation)
                        result_queue = multiprocessing.Queue()
                        if sys.argv[3] == 'summarysearch':
                            target = run_summarysearch
                        elif sys.argv[3] == 'rclsolve':
                            target = run_rclsolve
                        else:
                            raise ValueError(f'Unknown solver: {sys.argv[3]}')
                        process = multiprocessing.Process(
                            target=target, args=(query, dbInfo, result_queue))
                        process.start()
                        process.join(timeout=TIMEOUT)
                        if process.is_alive():
                            process.terminate()
                            process.join()
                            timed_out = True
                            break
                        runtime, objective_value = result_queue.get()
                        runtimes.append(runtime)
                        if sys.argv[3] == 'rclsolve':
                            if lp_bound == 0.0:
                                errors.append(0.0)
                            else:
                                errors.append(
                                    (lp_bound - objective_value) / lp_bound)

                    print('Relation:', relation)
                    file_name = 'Runs/IncVar/' + relation + '_' + str(sys.argv[3])
                    if timed_out:
                        with open(file_name, "w", encoding="utf-8") as f:
                            print(
                                f'Did not terminate within the allotted'
                                f' timeout of {TIMEOUT} seconds', file=f)
                    else:
                        print('Runtimes:', runtimes)
                        print('Errors:', errors)
                        with open(file_name, "w", encoding="utf-8") as f:
                            print('Runtime Mean:', round(np.mean(runtimes), 1), file=f)
                            print('Runtime Variances:', round(np.var(runtimes), 2), file=f)
                            if len(errors) > 0:
                                print('Error mean:', round(np.mean(errors), 2), file=f)
                                print('Error Variances:', round(np.var(errors), 2), file=f)
                    print('Results written to', file_name)

            elif sys.argv[1] == 'inc_tuples':
                for relation in inc_tuples_relations:
                    runtimes = []
                    errors = []
                    timed_out = False
                    if sys.argv[3] == 'sskr':
                        print('Getting LP Upper Bound.')
                        query.set_relation(relation)
                        lp_bound = get_lp_upper_bound(query, dbInfo)
                        print('LP Upper Bound =', lp_bound)
                    for _ in range(iterations):
                        query.set_relation(relation)
                        result_queue = multiprocessing.Queue()
                        if sys.argv[3] == 'summarysearch':
                            target = run_summarysearch
                        elif sys.argv[3] == 'rclsolve':
                            target = run_rclsolve
                        elif sys.argv[3] == 'sskr':
                            target = run_sskr
                        else:
                            raise ValueError(f'Unknown solver: {sys.argv[3]}')
                        process = multiprocessing.Process(
                            target=target, args=(query, dbInfo, result_queue))
                        process.start()
                        process.join(timeout=TIMEOUT)
                        if process.is_alive():
                            process.terminate()
                            process.join()
                            timed_out = True
                            break
                        runtime, objective_value = result_queue.get()
                        runtimes.append(runtime)
                        if sys.argv[3] == 'sskr':
                            if lp_bound == 0.0:
                                errors.append(0.0)
                            else:
                                errors.append(
                                    (lp_bound - objective_value) / lp_bound)

                    print('Relation:', relation)
                    file_name = 'Runs/IncTuples/' + relation + '_' + str(sys.argv[3])
                    if timed_out:
                        with open(file_name, "w", encoding="utf-8") as f:
                            print(
                                f'Did not terminate within the allotted'
                                f' timeout of {TIMEOUT} seconds', file=f)
                    else:
                        print('Runtimes:', runtimes)
                        print('Errors:', errors)
                        with open(file_name, "w", encoding="utf-8") as f:
                            print('Runtime Mean:', round(np.mean(runtimes), 1), file=f)
                            print('Runtime Variances:', round(np.var(runtimes), 2), file=f)
                            if len(errors) > 0:
                                print('Error mean:', round(np.mean(errors), 2), file=f)
                                print('Error Variances:', round(np.var(errors), 2), file=f)
                    print('Results written to', file_name)
        break
