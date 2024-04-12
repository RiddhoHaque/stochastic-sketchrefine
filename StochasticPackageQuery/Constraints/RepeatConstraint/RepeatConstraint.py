from StochasticPackageQuery.Constraints.Constraint import Constraint


class RepeatConstraint(Constraint):

    def __init__(self):
        self.__repetition_limit = 0
        self.__is_repetition_limit_set = False

    def is_repeat_constraint(self) -> bool:
        return True

    def is_repetition_limit_set(self) -> bool:
        return self.__is_repetition_limit_set

    def set_repetition_limit(self, repetition_limit : int) -> None:
        if repetition_limit < 0:
            raise ValueError()
        self.__repetition_limit = repetition_limit
        if not self.__is_repetition_limit_set:
            self.__is_repetition_limit_set = True

    def get_repetition_limit(self) -> int:
        if not self.__is_repetition_limit_set:
            raise Exception('Repetition limit not set')
        return self.__repetition_limit

    def add_digit_to_repetition_limit(self, digit: int)-> int:
        if digit < 0 or digit >= 10:
            raise ValueError('Digit must be within [0-9]')
        if not self.__is_repetition_limit_set:
            self.__is_repetition_limit_set = True
            self.__repetition_limit = digit
        else:
            self.__repetition_limit *= 10
            self.__repetition_limit += digit