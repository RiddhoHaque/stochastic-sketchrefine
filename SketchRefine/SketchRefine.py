import binpacking

from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from StochasticPackageQuery.Query import Query
from SketchRefine.Sketch import Sketch
from SketchRefine.Refine import Refine


class SketchRefine:

    def __init__(
        self, query: Query, dbInfo: DbInfo,
        is_lp_relaxation = False
    ):
        
        self.__sketch = Sketch(
            query=query,
            dbInfo=dbInfo,
            no_of_opt_scenarios=\
                Hyperparameters.INIT_NO_OF_SCENARIOS,
            is_lp_relaxation=is_lp_relaxation
        )

        self.__partition_sizes = \
            self.__sketch.get_partition_sizes()
        
        self.__max_no_of_duplicates = \
            self.__sketch.get_max_no_of_duplicates()
        
        self.__query = query
        self.__dbInfo = dbInfo
                
    
    def solve(self):
        sketch_package, sketch_objective_value,\
            no_of_optimization_scenarios = \
                self.__sketch.solve()
        
        #print('Sketch Package:', sketch_package)
        sizes = dict()
        for partition_id in sketch_package:
            sizes[partition_id] = self.__partition_sizes[
                partition_id]

        bins = binpacking.to_constant_volume(
            sizes, Hyperparameters.SIZE_THRESHOLD)
        
        partition_groups = []

        for bin in bins:
            partition_groups.append([])
            for key in bin.keys():
                partition_groups[-1].append(key)
        
        print('Partition groups:', partition_groups)
        refine = Refine(
            partition_groups,
            no_of_optimization_scenarios,
            self.__max_no_of_duplicates,
            sketch_objective_value,
            sketch_package,
            self.__query,
            self.__dbInfo
        )

        #refine.solve()
        return sketch_package, sketch_objective_value
