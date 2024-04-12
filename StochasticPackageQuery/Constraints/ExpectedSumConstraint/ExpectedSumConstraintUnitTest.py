import unittest
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint


class ExpectedSumConstraintUnitTest(unittest.TestCase):

        def test_initial_conditions(self):
            expected_sum_constraint = ExpectedSumConstraint()
            self.assertFalse(expected_sum_constraint.is_deterministic_constraint())
            self.assertTrue(expected_sum_constraint.is_expected_sum_constraint())
            self.assertFalse(expected_sum_constraint.is_sum_limit_set())
            self.assertFalse(expected_sum_constraint.is_inequality_sign_set())

        def test_sum_limit_consistency(self):
            expected_sum_constraint = ExpectedSumConstraint()
            expected_sum_constraint.set_sum_limit(-5.5)
            self.assertAlmostEqual(expected_sum_constraint.get_sum_limit(), -5.5)

        def main(self):
            self.test_initial_conditions()
            self.test_sum_limit_consistency()