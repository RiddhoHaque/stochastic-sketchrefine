import unittest
from StochasticPackageQuery.Constraints.Constraint import Constraint


class ConstraintUnitTest(unittest.TestCase):

    def test_constraint_types(self):
        constraint = Constraint()
        self.assertFalse(constraint.is_var_constraint())
        self.assertFalse(constraint.is_deterministic_constraint())
        self.assertFalse(constraint.is_repeat_constraint())
        self.assertFalse(constraint.is_package_size_constraint())
        self.assertFalse(constraint.is_cvar_constraint())
        self.assertFalse(constraint.is_risk_constraint())
        self.assertFalse(constraint.is_expected_sum_constraint())

    def main(self):
        self.test_constraint_types()