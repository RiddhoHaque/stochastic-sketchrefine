import time


class OptimizationMetrics:

    def __init__(
        self,
        algorithm_name: str,
        linear_relaxation: bool,
    ) -> None:
        self.__algorithm_name = \
            algorithm_name
        self.__linear_relaxation = \
            linear_relaxation
        self.__runtime = 0
        self.__objective_value = 0
        self.__optimizer_starting_time = None
        self.__optimizer_runtime = 0
        self.__number_of_optimization_calls = 0
        self.__starting_time = None

    def start_execution(self):
       self.__start_runtime = time.time()

    def end_execution(
        self, objective_value
    ):
        self.__runtime = \
            time.time() - self.__start_runtime
        self.__objective_value = \
            objective_value
    
    def start_optimizer(self):
        self.__optimizer_starting_time = \
            time.time()
        
    def end_optimizer(self):
        assert self.__optimizer_starting_time\
            is not None
        self.__optimizer_runtime += time.time()\
            - self.__optimizer_starting_time
        self.__number_of_optimization_calls += 1
    
    def log(self):
        print('Algorithm:',
              self.__algorithm_name)
        print('Linear Relaxation:',
              self.__linear_relaxation)
        print('Runtime:',
              self.__runtime)
        print('Objective Value:',
              self.__objective_value)
        print('Number of optimization calls:', 
              self.__number_of_optimization_calls)
        print('Total Optimizer Runtime:',
              self.__optimizer_runtime)
