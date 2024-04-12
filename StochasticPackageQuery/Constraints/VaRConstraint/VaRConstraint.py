from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraint import DeterministicConstraint


class VaRConstraint(DeterministicConstraint):

    def __init__(self):
        super().__init__()
        self.__is_probability_threshold_set = False
        self.__probability_threshold = 0.0
        self.__cached_probability_string = ''

    def is_risk_constraint(self) -> bool:
        return True

    def is_var_constraint(self) -> bool:
        return True

    def is_probability_threshold_set(self) -> bool:
        return self.__is_probability_threshold_set

    def set_probability_threshold(self, probability_threshold: float):
        if probability_threshold < 0.0 or probability_threshold > 1.0:
            raise Exception
        self.__probability_threshold = probability_threshold
        self.__is_probability_threshold_set = True
        self.__cached_probability_string = ''

    def add_character_to_probability_threshold(self, char: chr):
        self.__cached_probability_string += char
        try:
            self.__probability_threshold = float(self.__cached_probability_string)
            if self.__probability_threshold < 0.0 or self.__probability_threshold > 1.0:
                raise Exception
            self.__is_probability_threshold_set = True
        except TypeError:
            ...

    def get_probability_threshold(self):
        if not self.__is_probability_threshold_set:
            raise Exception
        return self.__probability_threshold
