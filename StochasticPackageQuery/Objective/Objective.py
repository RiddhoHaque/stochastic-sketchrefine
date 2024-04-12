from Utils.ObjectiveType import ObjectiveType
from Utils.Stochasticity import Stochasticity


class Objective:

    def __init__(self):
        self.__is_objective_type_set = False
        self.__objective_type = ObjectiveType.MAXIMIZATION
        self.__attribute_name = ''
        self.__is_stochasticity_set = False
        self.__stochasticity = Stochasticity.STOCHASTIC

    def is_objective_type_set(self) -> bool:
        return self.__is_objective_type_set

    def is_stochasticity_set(self) -> bool:
        return self.__is_stochasticity_set

    def get_objective_type(self) -> int:
        if not self.__is_objective_type_set:
            raise Exception
        return self.__objective_type

    def get_attribute_name(self) -> str:
        return self.__attribute_name

    def get_stochasticity(self) -> str:
        if not self.__is_stochasticity_set:
            raise Exception
        return self.__stochasticity

    def set_objective_type(self, is_maximization: bool):
        self.__is_objective_type_set = True
        if is_maximization:
            self.__objective_type = ObjectiveType.MAXIMIZATION
        else:
            self.__objective_type = ObjectiveType.MINIMIZATION

    def set_stochasticity(self, is_stochastic: bool):
        self.__is_stochasticity_set = True
        if is_stochastic:
            self.__stochasticity = Stochasticity.STOCHASTIC
        else:
            self.__stochasticity = Stochasticity.DETERMINISTIC

    def set_attribute_name(self, attribute_name: str):
        self.__attribute_name = attribute_name

    def add_character_to_attribute_name(self, char: chr):
        self.__attribute_name += char