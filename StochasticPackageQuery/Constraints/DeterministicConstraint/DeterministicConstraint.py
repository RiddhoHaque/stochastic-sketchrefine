from StochasticPackageQuery.Constraints.Constraint import Constraint
from Utils.RelationalOperators import RelationalOperators


class DeterministicConstraint(Constraint):

    def __init__(self):
        self.__attribute_name = ''
        self.__is_inequality_sign_set = False
        self.__inequality_sign = RelationalOperators.LESS_THAN_OR_EQUAL_TO
        self.__is_sum_limit_set = False
        self.__sum_limit = 0.0
        self.__cached_sum_limit_string = ''

    def is_deterministic_constraint(self) -> bool:
        return True

    def is_inequality_sign_set(self) -> bool:
        return self.__is_inequality_sign_set

    def is_sum_limit_set(self) -> bool:
        return self.__is_sum_limit_set

    def get_inequality_sign(self) -> bool:
        if not self.__is_inequality_sign_set:
            raise Exception('Inequality sign not set')
        return self.__inequality_sign

    def get_sum_limit(self):
        if not self.__is_sum_limit_set:
            raise Exception('Sum limit is not set')
        return self.__sum_limit

    def set_inequality_sign(self, char : chr):
        if char != '<' and char != '>' and char != '=':
            raise Exception('Unrecognized char for inequality sign')
        if self.__is_inequality_sign_set:
            raise Exception('Inequality sign is already set')
        self.__is_inequality_sign_set = True
        if char == '<':
            self.__inequality_sign = RelationalOperators.LESS_THAN_OR_EQUAL_TO
        elif char == '>':
            self.__inequality_sign = RelationalOperators.GREATER_THAN_OR_EQUAL_TO
        else:
            self.__inequality_sign = RelationalOperators.EQUALS

    def set_sum_limit(self, sum_limit: float):
        self.__is_sum_limit_set = True
        self.__sum_limit = sum_limit
        self.__cached_sum_limit_string = ''

    def add_character_to_sum_limit(self, char: chr):
        self.__cached_sum_limit_string = self.__cached_sum_limit_string + char
        try:
            self.__sum_limit = float(self.__cached_sum_limit_string)
            self.__is_sum_limit_set = True
        except:
            ...

    def set_attribute_name(self, attribute_name: str):
        self.__attribute_name = attribute_name

    def add_character_to_attribute_name(self, char: chr):
        self.__attribute_name += char

    def get_attribute_name(self) -> str:
        return self.__attribute_name