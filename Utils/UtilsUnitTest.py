import unittest
from Utils.ObjectiveType import ObjectiveType
from Utils.RelationalOperators import RelationalOperators
from Utils.Stochasticity import Stochasticity


class UtilsUnitTest(unittest.TestCase):

    def test_objective_type_uniqueness(self):
        self.assertFalse(ObjectiveType.MAXIMIZATION == ObjectiveType.MINIMIZATION)

    def test_relational_operator_uniqueness(self):
        self.assertFalse(RelationalOperators.LESS_THAN_OR_EQUAL_TO == RelationalOperators.GREATER_THAN_OR_EQUAL_TO)
        self.assertFalse(RelationalOperators.EQUALS == RelationalOperators.GREATER_THAN_OR_EQUAL_TO)
        self.assertFalse(RelationalOperators.LESS_THAN_OR_EQUAL_TO == RelationalOperators.EQUALS)

    def test_stochasticity_uniqueness(self):
        self.assertFalse(Stochasticity.STOCHASTIC == Stochasticity.DETERMINISTIC)

    def main(self):
        self.test_objective_type_uniqueness()
        self.test_relational_operator_uniqueness()
        self.test_stochasticity_uniqueness()