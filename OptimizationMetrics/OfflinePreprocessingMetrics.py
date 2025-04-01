import time

class OfflinePreprocessingMetrics:

    def __init__(self):
        self.__scenario_generation_starting_time = None
        self.__scenario_generation_ending_time = None
        self.__partitioning_starting_time = None
        self.__partitioning_ending_time = None
        self.__partitioning_table_creation_starting_time = None
        self.__partitioning_table_creation_ending_time = None
        self.__representative_selection_starting_time = None
        self.__representative_selection_ending_time = None
        self.__representative_histogram_starting_time = None
        self.__representative_histogram_ending_time = None
        self.__correlation_estimation_starting_time = None
        self.__correlation_estimation_ending_time = None


    def start_scenario_generation(self):
        self.__scenario_generation_starting_time = \
            time.time()

    def end_scenario_generation(self):
        self.__scenario_generation_ending_time = \
            time.time()

    def start_partitioning(self):
        self.__partitioning_starting_time = \
            time.time()

    def end_partitioning(self):
        self.__partitioning_ending_time = \
            time.time()

    def start_partitioning_table_creation(self):
        self.__partitioning_table_creation_starting_time = \
            time.time()

    def end_partitioning_table_creation(self):
        self.__partitioning_table_creation_ending_time = \
            time.time()
    
    def start_representative_selection(self):
        self.__representative_selection_starting_time = \
            time.time()
        
    def end_representative_selection(self):
        self.__representative_selection_ending_time = \
            time.time()
    
    def start_representative_histogram_estimation(self):
        self.__representative_histogram_starting_time = \
            time.time()
    
    def end_representative_histogram_estimation(self):
        self.__representative_histogram_ending_time = \
            time.time()
    
    def start_required_correlation_estimation(self):
        self.__correlation_estimation_starting_time = \
            time.time()
        
    def end_required_correlation_estimation(self):
        self.__correlation_estimation_ending_time = \
            time.time()
        
    def log_performance(self):
        print('Scenario Generation time:',
            self.__scenario_generation_ending_time - \
                self.__scenario_generation_starting_time)
        
        print('Partitioning time:',
            self.__partitioning_ending_time - \
                self.__partitioning_starting_time)
        
        print('Partitioning Table Creation time:',
            self.__partitioning_table_creation_ending_time - \
                self.__partitioning_table_creation_starting_time)
        
        print('Representative Selection Starting time:',
            self.__representative_selection_ending_time - \
                self.__representative_selection_starting_time)
        
        if self.__representative_histogram_ending_time is not None:
            print('Representative histogram estimation time:',
                self.__representative_histogram_ending_time - \
                    self.__representative_histogram_starting_time)
        
            print('Correlation estimation time:',
                self.__correlation_estimation_ending_time - \
                    self.__correlation_estimation_starting_time)
        
            print('Total time:', self.__correlation_estimation_ending_time -\
                    self.__scenario_generation_starting_time)
        
        else:
            print('Total time:', self.__representative_selection_ending_time -\
                    self.__representative_selection_starting_time)