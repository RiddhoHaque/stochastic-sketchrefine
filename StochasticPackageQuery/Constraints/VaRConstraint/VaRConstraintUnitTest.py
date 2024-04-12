import unittest
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraint import VaRConstraint


class VaRConstraintUnitTest(unittest.TestCase):

    def test_initial_conditions(self):
        var_constraint = VaRConstraint()
        self.assertTrue(var_constraint.is_risk_constraint())
        self.assertTrue(var_constraint.is_var_constraint())
        self.assertFalse(var_constraint.is_sum_limit_set())
        self.assertFalse(var_constraint.is_inequality_sign_set())
        self.assertFalse(var_constraint.is_probability_threshold_set())

    def test_probability_threshold_consistency(self):
        var_constraint = VaRConstraint()
        var_constraint.set_probability_threshold(0.95)
        self.assertEqual(var_constraint.get_probability_threshold(), 0.95)

        var_constraint = VaRConstraint()
        var_constraint.set_probability_threshold(1.0)
        self.assertEqual(var_constraint.get_probability_threshold(), 1.0)

        var_constraint = VaRConstraint()
        var_constraint.set_probability_threshold(0.0)
        self.assertEqual(var_constraint.get_probability_threshold(), 0.0)

        with self.assertRaises(Exception):
            var_constraint.set_probability_threshold(-0.5)

        with self.assertRaises(Exception):
            var_constraint.set_probability_threshold(1.3)

    def test_setting_probability_threshold_digit_by_digit(self):
        var_constraint = VaRConstraint()

        var_constraint.add_character_to_probability_threshold('0')
        var_constraint.add_character_to_probability_threshold('.')
        var_constraint.add_character_to_probability_threshold('9')
        var_constraint.add_character_to_probability_threshold('5')
        self.assertTrue(var_constraint.is_probability_threshold_set())
        self.assertAlmostEqual(var_constraint.get_probability_threshold(), 0.95)

        with self.assertRaises(Exception):
            var_constraint = VaRConstraint()
            var_constraint.add_character_to_probability_threshold('-')
            var_constraint.add_character_to_probability_threshold('0')
            var_constraint.add_character_to_probability_threshold('.')
            var_constraint.add_character_to_probability_threshold('0')
            var_constraint.add_character_to_probability_threshold('5')

        with self.assertRaises(Exception):
            var_constraint = VaRConstraint()
            var_constraint.add_character_to_probability_threshold('1')
            var_constraint.add_character_to_probability_threshold('.')
            var_constraint.add_character_to_probability_threshold('0')
            var_constraint.add_character_to_probability_threshold('5')

    def test_get_probability_threshold_before_setting_it(self):
        var_constraint = VaRConstraint()
        with self.assertRaises(Exception):
            var_constraint.get_probability_threshold()

    def main(self):
        self.test_initial_conditions()
        self.test_probability_threshold_consistency()
        self.test_setting_probability_threshold_digit_by_digit()
        self.test_get_probability_threshold_before_setting_it()