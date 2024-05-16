from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from Utils.RelationalOperators import RelationalOperators
from Utils.TailType import TailType


class CVaRConstraint(ExpectedSumConstraint):

    def __init__(self):
        super().__init__()
        self.__is_tail_type_set = False
        self.__tail_type = TailType.LOWEST
        self.__is_percentage_of_scenarios_set = False
        self.__percentage_of_scenarios = 0.0
        self.__cached_percentage_string = ''
    
    def initialize_from_expected_sum_constraint(
            self, expected_sum_constraint: ExpectedSumConstraint
        ):
        super().set_attribute_name(expected_sum_constraint.get_attribute_name())
        if expected_sum_constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            super().set_inequality_sign('>')
        elif expected_sum_constraint.get_inequality_sign() == \
            RelationalOperators.LESS_THAN_OR_EQUAL_TO:
            super().set_inequality_sign('<')
        else:
            super().set_inequality_sign('=')
        
        super().set_sum_limit(expected_sum_constraint.get_sum_limit())
        self.__is_tail_type_set = False
        self.__is_percentage_of_scenarios_set = False
        self.__percentage_of_scenarios = 0.0
        self.__cached_percentage_string = ''
    
    def is_risk_constraint(self) -> bool:
        return True

    def is_cvar_constraint(self) -> bool:
        return True
    
    def is_expected_sum_constraint(self) -> bool:
        return False

    def is_percentage_of_scenarios_set(self) -> bool:
        return self.__is_percentage_of_scenarios_set
    
    def set_percentage_of_scenarios(self, percentage_of_scenarios: float):
        if percentage_of_scenarios <= 0.0 or percentage_of_scenarios > 100.0:
            raise Exception
        self.__percentage_of_scenarios = percentage_of_scenarios
        self.__is_percentage_of_scenarios_set = True
        self.__cached_percentage_string = ''
    
    def add_character_to_percentage_of_scenarios(self, char: chr):
        self.__cached_percentage_string += char
        try:
            self.__percentage_of_scenarios = float(self.__cached_percentage_string)
            if self.__percentage_of_scenarios <= 0.0 or self.__percentage_of_scenarios > 100.0:
                raise Exception
            self.__is_percentage_of_scenarios_set = True
        except TypeError:
            ...
    
    def get_percentage_of_scenarios(self):
        if not self.__is_percentage_of_scenarios_set:
            raise Exception
        return self.__percentage_of_scenarios

    def is_tail_type_set(self) -> bool:
        return self.__is_tail_type_set
    
    def get_tail_type(self) -> bool:
        if not self.__is_tail_type_set:
            raise Exception
        return self.__tail_type

    def set_tail_type(self, char: chr) -> bool:
        if char == 'h':
            self.__is_tail_type_set = True
            self.__tail_type = TailType.HIGHEST
        elif char == 'l':
            self.__is_tail_type_set = True
            self.__tail_type = TailType.LOWEST
        else:
            raise Exception