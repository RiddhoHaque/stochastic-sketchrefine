from StochasticPackageQuery.Constraints.Constraint import Constraint
from StochasticPackageQuery.Constraints.RepeatConstraint.RepeatConstraint import RepeatConstraint
from StochasticPackageQuery.Constraints.PackageSizeConstraint.PackageSizeConstraint import PackageSizeConstraint
from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraint import DeterministicConstraint
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraint import VaRConstraint
from StochasticPackageQuery.Objective.Objective import Objective


class Query:

    def __init__(self):
        self.__projected_attributes = ''
        self.__package_alias = ''
        self.__relation = ''
        self.__relation_alias = ''
        self.__base_predicate = ''
        self.__constraints = []
        self.__objective = Objective()

    def set_projected_attributes(self, projected_attributes: str):
        self.__projected_attributes = projected_attributes

    def get_projected_attributes(self):
        return self.__projected_attributes

    def add_character_to_projected_attributes(self, char: chr):
        self.__projected_attributes += char

    def set_package_alias(self, package_alias: str):
        self.__package_alias = package_alias

    def get_package_alias(self):
        return self.__package_alias

    def add_character_to_package_alias(self, char: chr):
        self.__package_alias += char
    
    def set_base_predicate(self, base_predicate: str):
        self.__base_predicate = base_predicate

    def get_base_predicate(self) -> str:
        return self.__base_predicate

    def add_character_to_base_predicate(self, char: chr):
        self.__base_predicate += char

    def set_relation(self, relation: str):
        self.__relation = relation

    def get_relation(self):
        return self.__relation

    def add_character_to_relation(self, char: chr):
        self.__relation += char

    def set_relation_alias(self, relation_alias: str):
        self.__relation_alias = relation_alias

    def get_relation_alias(self):
        return self.__relation_alias

    def add_character_to_relation_alias(self, char: chr):
        self.__relation_alias += char

    def add_constraint(self, constraint: Constraint):
        self.__constraints.append(constraint)

    def add_repeat_constraint(self):
        self.__constraints.append(RepeatConstraint())

    def add_digit_to_repeat_constraint(self, digit: int):
        if len(self.__constraints) < 1 or not self.__constraints[-1].is_repeat_constraint():
            raise Exception
        self.__constraints[-1].add_digit_to_repetition_limit(digit)

    def add_package_size_constraint(self):
        self.__constraints.append(PackageSizeConstraint())

    def add_digit_to_package_size_constraint(self, digit: int):
        if len(self.__constraints) < 1 or not self.__constraints[-1].is_package_size_constraint():
            raise Exception
        self.__constraints[-1].add_digit_to_package_size_limit(digit)

    def add_deterministic_constraint(self):
        self.__constraints.append(DeterministicConstraint())

    def add_expected_sum_constraint(self):
        self.__constraints.append(ExpectedSumConstraint())

    def add_var_constraint(self):
        self.__constraints.append(VaRConstraint())

    def add_character_to_attribute_name(self, char: chr):
        if len(self.__constraints) < 1 or (not self.__constraints[-1].is_deterministic_constraint() and
                                           not self.__constraints[-1].is_expected_sum_constraint() and
                                           not self.__constraints[-1].is_var_constraint()):
            raise Exception
        self.__constraints[-1].add_character_to_attribute_name(char)

    def set_constraint_inequality_sign(self, char: chr):
        if len(self.__constraints) < 1 or (not self.__constraints[-1].is_package_size_constraint() and
                                           not self.__constraints[-1].is_deterministic_constraint() and
                                           not self.__constraints[-1].is_expected_sum_constraint() and
                                           not self.__constraints[-1].is_var_constraint()):
            raise Exception
        self.__constraints[-1].set_inequality_sign(char)

    def add_character_to_constraint_sum_limit(self, char: chr):
        if len(self.__constraints) < 1 or (not self.__constraints[-1].is_deterministic_constraint() and
                                           not self.__constraints[-1].is_expected_sum_constraint() and
                                           not self.__constraints[-1].is_var_constraint()):
            raise Exception
        self.__constraints[-1].add_character_to_sum_limit(char)

    def convert_final_deterministic_constraint_to_var_constraint(self):
        if len(self.__constraints) < 1 or not self.__constraints[-1].is_deterministic_constraint():
            raise Exception
        deterministic_constraint = self.__constraints[-1]
        var_constraint = VaRConstraint()
        var_constraint.initialize_from_deterministic_constraint(deterministic_constraint)
        self.__constraints[-1] = var_constraint
    
    def add_character_to_constraint_probability_threshold(self, char: chr):
        if len(self.__constraints) < 1 or not self.__constraints[-1].is_var_constraint():
            raise Exception
        self.__constraints[-1].add_character_to_probability_threshold(char)

    def get_constraints(self):
        return self.__constraints

    def set_objective(self, objective: Objective):
        self.__objective = objective

    def set_objective_type(self, is_maximization: bool):
        self.__objective.set_objective_type(is_maximization)

    def set_objective_stochasticity(self, is_stochastic: bool):
        self.__objective.set_stochasticity(is_stochastic)

    def add_character_to_objective_attribute(self, char: chr):
        self.__objective.add_character_to_attribute_name(char)

    def get_objective(self):
        return self.__objective