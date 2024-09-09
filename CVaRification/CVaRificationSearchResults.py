class CVaRificationSearchResults:

    def __init__(self):
        self.__needs_more_scenarios = False
        self.__found_appropriate_package = False
        self.__package = None
        self.__objective_upper_bound = 0
        self.__cvar_thresholds = []
        self.__scenarios_to_consider = []
        self.__validation_objective_value = 0.0

    def get_needs_more_scenarios(self):
        return self.__needs_more_scenarios
    
    def set_needs_more_scenarios(
        self, needs_scenarios):
        self.__needs_more_scenarios =\
            needs_scenarios
    
    def get_found_appropriate_package(self):
        return self.__found_appropriate_package
    
    def set_found_appropriate_package(
        self, found_appropriate_package
    ):
        self.__found_appropriate_package = \
            found_appropriate_package
    
    def get_package(self):
        return self.__package
    
    def set_package(self, package):
        self.__package = package
    
    def get_objective_upper_bound(self):
        return self.__objective_upper_bound
    
    def set_objective_upper_bound(
        self, objective_upper_bound):
        self.__objective_upper_bound = \
            objective_upper_bound
        
    def get_cvar_thresholds(self):
        return self.__cvar_thresholds
    
    def set_cvar_thresholds(
        self, cvar_thresholds
    ):
        self.__cvar_thresholds = \
            cvar_thresholds
    
    def get_scenarios_to_consider(self):
        return self.__scenarios_to_consider
    
    def set_scenarios_to_consider(
        self, scenarios_to_consider):
        self.__scenarios_to_consider =\
            scenarios_to_consider
    
    def get_objective_value(self):
        return self.__validation_objective_value
    
    def set_objective_value(
            self, objective_value):
        self.__validation_objective_value =\
            objective_value