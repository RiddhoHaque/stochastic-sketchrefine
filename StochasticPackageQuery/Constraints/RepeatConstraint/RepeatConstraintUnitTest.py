import unittest
from StochasticPackageQuery.Constraints.RepeatConstraint.RepeatConstraint import RepeatConstraint


class RepeatConstraintUnitTest(unittest.TestCase):

    def test_initial_conditions(self):
        repeat_constraint = RepeatConstraint()
        self.assertTrue(repeat_constraint.is_repeat_constraint())
        self.assertFalse(repeat_constraint.is_repetition_limit_set())

    def test_repetition_limit_consistency(self):

        repeat_constraint = RepeatConstraint()
        repeat_constraint.set_repetition_limit(0)
        self.assertEqual(repeat_constraint.get_repetition_limit(), 0)
        self.assertTrue(repeat_constraint.is_repetition_limit_set())

        repeat_constraint.set_repetition_limit(10)
        self.assertEqual(repeat_constraint.get_repetition_limit(), 10)
        self.assertTrue(repeat_constraint.is_repetition_limit_set())

        repeat_constraint.set_repetition_limit(5)
        self.assertEqual(repeat_constraint.get_repetition_limit(), 5)
        self.assertTrue(repeat_constraint.is_repetition_limit_set())

        with self.assertRaises(ValueError):
            repeat_constraint.set_repetition_limit(-1)

    def test_setting_repetition_limit_digit_by_digit(self):
        repeat_constraint = RepeatConstraint()

        repeat_constraint.add_digit_to_repetition_limit(1)
        self.assertTrue(repeat_constraint.is_repetition_limit_set())
        self.assertEqual(repeat_constraint.get_repetition_limit(), 1)

        repeat_constraint.add_digit_to_repetition_limit(0)
        self.assertTrue(repeat_constraint.is_repetition_limit_set())
        self.assertEqual(repeat_constraint.get_repetition_limit(), 10)

        repeat_constraint.add_digit_to_repetition_limit(2)
        self.assertTrue(repeat_constraint.is_repetition_limit_set())
        self.assertEqual(repeat_constraint.get_repetition_limit(), 102)

        with self.assertRaises(ValueError):
            repeat_constraint.add_digit_to_repetition_limit(-1)

        with self.assertRaises(ValueError):
            repeat_constraint.add_digit_to_repetition_limit(10)

    def test_get_repetition_limit_before_setting_it(self):
        repeat_constraint = RepeatConstraint()
        with self.assertRaises(Exception):
            repeat_constraint.get_repetition_limit()
    def main(self):
        self.test_initial_conditions()
        self.test_repetition_limit_consistency()
        self.test_setting_repetition_limit_digit_by_digit()
        self.test_get_repetition_limit_before_setting_it()