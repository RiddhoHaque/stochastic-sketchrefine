from StochasticPackageQuery.Constraints.Constraint import Constraint
from Utils.RelationalOperators import RelationalOperators


class PackageSizeConstraint(Constraint):

    def __init__(self):
        self.__package_size_limit = 0
        self.__inequality_sign = '<='
        self.__is_package_size_limit_set = RelationalOperators.LESS_THAN_OR_EQUAL_TO
        self.__is_inequality_sign_set = False

    def is_package_size_constraint(self) -> bool:
        return True

    def is_package_size_limit_set(self) -> bool:
        return self.__is_package_size_limit_set

    def set_package_size_limit(self, package_size_limit : int) -> None:
        if package_size_limit < 0:
            raise ValueError()
        self.__package_size_limit = package_size_limit
        if not self.__is_package_size_limit_set:
            self.__is_package_size_limit_set = True

    def set_inequality_sign(self, char: chr):
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

    def is_inequality_sign_set(self):
        return self.__is_inequality_sign_set

    def get_package_size_limit(self) -> int:
        if not self.__is_package_size_limit_set:
            raise Exception('Package size limit not set')
        return self.__package_size_limit

    def get_inequality_sign(self) -> RelationalOperators:
        if not self.__is_inequality_sign_set:
            raise Exception('Inequality sign not set')
        return self.__inequality_sign

    def add_digit_to_package_size_limit(self, digit: int)-> int:
        if digit < 0 or digit >= 10:
            raise ValueError('Digit must be within [0-9]')
        if not self.__is_package_size_limit_set:
            self.__is_package_size_limit_set = True
            self.__package_size_limit = digit
        else:
            self.__package_size_limit *= 10
            self.__package_size_limit += digit