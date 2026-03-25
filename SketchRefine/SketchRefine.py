import binpacking

from CVaRification.RCLSolve import RCLSolve
from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OptimizationMetrics.OptimizationMetrics import OptimizationMetrics
from PgConnection.PgConnection import PgConnection
from StochasticPackageQuery.Query import Query
from SketchRefine.Sketch import Sketch
from SketchRefine.Refine import Refine


class SketchRefine:

    def __init__(
        self, query: Query, dbInfo: DbInfo,
        is_lp_relaxation = False
    ):
        self.__partition_sizes = []
        self.__max_no_of_duplicates = []
        self.__query = query
        self.__dbInfo = dbInfo
        self.__is_lp_relaxation = is_lp_relaxation
        self.__metrics = OptimizationMetrics(
            'SketchRefine', is_lp_relaxation)
                
    
    def __get_relation_size(self):
        relation = self.__query.get_relation()
        base_predicate = self.__query.get_base_predicate()
        sql_query = "SELECT COUNT(*) FROM " + relation
        if base_predicate and base_predicate != '1=1':
            sql_query += " WHERE " + base_predicate
        PgConnection.Execute(sql_query)
        return int(PgConnection.Fetch()[0][0])
    

    def solve(self):
        self.__metrics.start_execution()
        print('Relation Size:', self.__get_relation_size())
        print('Size threshold:', Hyperparameters.SIZE_THRESHOLD)
        if self.__get_relation_size() <= \
            Hyperparameters.SIZE_THRESHOLD:
            print('Relation size small enough to apply RCLSolve directly')
            rclsolver = RCLSolve(
                query=self.__query,
                linear_relaxation=self.__is_lp_relaxation,
                dbInfo=self.__dbInfo,
                init_no_of_scenarios=\
                    Hyperparameters.INIT_NO_OF_SCENARIOS,
                no_of_validation_scenarios=\
                    Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
                approximation_bound=\
                    Hyperparameters.APPROXIMATION_BOUND,
                sampling_tolerance=\
                    Hyperparameters.SAMPLING_TOLERANCE,
                bisection_threshold=\
                    Hyperparameters.BISECTION_THRESHOLD,
                max_opt_scenarios=\
                    Hyperparameters.MAX_OPT_SCENARIOS_IN_PRACTICE
            )
            result = rclsolver.solve()
            m = rclsolver.get_metrics()
            self.__metrics.add_optimizer_metrics(
                m.get_optimizer_runtime(),
                m.get_number_of_optimization_calls()
            )
            if result is None:
                self.__metrics.end_execution(0.0, 0)
                return None, 0.0
            self.__metrics.end_execution(result[1], 0)
            return result
        
        print('Initiating Sketch')
        sketch = Sketch(
            query=self.__query,
            dbInfo=self.__dbInfo,
            no_of_opt_scenarios=\
                Hyperparameters.INIT_NO_OF_SCENARIOS,
            is_lp_relaxation=self.__is_lp_relaxation
        )
        print('Sketch Initialized')

        self.__partition_sizes = \
            sketch.get_partition_sizes()

        self.__max_no_of_duplicates = \
            sketch.get_max_no_of_duplicates()

        print('Solving Sketch')
        sketch_package, sketch_objective_value,\
            no_of_optimization_scenarios = \
                sketch.solve()

        if sketch_package is None:
            sketch_m = sketch.get_metrics()
            self.__metrics.add_optimizer_metrics(
                sketch_m.get_optimizer_runtime(),
                sketch_m.get_number_of_optimization_calls()
            )
            self.__metrics.end_execution(0.0, 0)
            return None, 0.0

        print('Sketch Package:', sketch_package)
        result_package, result_objective = \
            self.__try_refine(
                sketch_package, sketch_objective_value,
                no_of_optimization_scenarios
            )

        if result_package is None:
            print('Sketch package did not refine — trying patch')

            # Phase 1: bump correlations if applicable
            if self.__dbInfo.has_inter_tuple_correlations():
                any_increased = sketch.bump_correlations(
                    list(sketch_package.keys()))
                if any_increased:
                    new_pkg, new_obj, new_scenarios = \
                        sketch.re_solve()
                    if new_pkg is not None:
                        result_package, result_objective = \
                            self.__try_refine(
                                new_pkg, new_obj, new_scenarios)

            # Phase 2: multiplicity upper bounds on each partition
            if result_package is None:
                for pid in sorted(sketch_package.keys()):
                    ub = int(sketch_package[pid]) - 1
                    print('Trying with upper bound', ub,
                          'on partition', pid)
                    new_pkg, new_obj, new_scenarios = \
                        sketch.re_solve(pid, ub)
                    if new_pkg is None:
                        continue
                    result_package, result_objective = \
                        self.__try_refine(
                            new_pkg, new_obj, new_scenarios)
                    if result_package is not None:
                        break

        if result_package is None:
            print('No alternative sketch package could be refined')

        sketch_m = sketch.get_metrics()
        self.__metrics.add_optimizer_metrics(
            sketch_m.get_optimizer_runtime(),
            sketch_m.get_number_of_optimization_calls()
        )
        self.__metrics.end_execution(result_objective, 0)
        return result_package, result_objective

    def __try_refine(
        self, sketch_package, sketch_objective_value,
        no_of_optimization_scenarios
    ):
        if sketch_package is None:
            return None, 0.0
        sizes = {}
        for partition_id in sketch_package:
            sizes[partition_id] = \
                self.__partition_sizes[partition_id]
        bins = binpacking.to_constant_volume(
            sizes, Hyperparameters.SIZE_THRESHOLD)
        partition_groups = []
        for bin in bins:
            partition_groups.append([])
            for key in bin.keys():
                partition_groups[-1].append(key)
        refine = Refine(
            partition_groups,
            no_of_optimization_scenarios,
            self.__max_no_of_duplicates,
            sketch_objective_value,
            sketch_package,
            self.__query,
            self.__dbInfo,
            self.__is_lp_relaxation
        )
        result = refine.solve()
        self.__metrics.add_optimizer_metrics(
            refine.get_optimizer_runtime(),
            refine.get_number_of_optimization_calls()
        )
        return result

    def get_metrics(self):
        return self.__metrics
