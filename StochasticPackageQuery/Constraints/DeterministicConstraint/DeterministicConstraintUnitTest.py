import unittest
from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraint import DeterministicConstraint
from Utils.RelationalOperators import RelationalOperators

class DeterministicConstraintUnitTest(unittest.TestCase):

        def test_initial_conditions(self):
            deterministic_constraint = DeterministicConstraint()
            self.assertTrue(deterministic_constraint.is_deterministic_constraint())
            self.assertFalse(deterministic_constraint.is_sum_limit_set())
            self.assertFalse(deterministic_constraint.is_inequality_sign_set())

        def test_sum_limit_consistency(self):
            deterministic_constraint = DeterministicConstraint()
            deterministic_constraint.set_sum_limit(5.5)
            self.assertEqual(deterministic_constraint.get_sum_limit(), 5.5)

            deterministic_constraint = DeterministicConstraint()
            deterministic_constraint.set_sum_limit(-10.3)
            self.assertEqual(deterministic_constraint.get_sum_limit(), -10.3)

        def test_inequality_sign_consistency(self):
            deterministic_constraint_1 = DeterministicConstraint()
            deterministic_constraint_2 = DeterministicConstraint()
            deterministic_constraint_3 = DeterministicConstraint()

            deterministic_constraint_1.set_inequality_sign('<')
            self.assertEqual(deterministic_constraint_1.get_inequality_sign(),
                             RelationalOperators.LESS_THAN_OR_EQUAL_TO)

            deterministic_constraint_2.set_inequality_sign('>')
            self.assertEqual(deterministic_constraint_2.get_inequality_sign(),
                             RelationalOperators.GREATER_THAN_OR_EQUAL_TO)

            deterministic_constraint_3.set_inequality_sign('=')
            self.assertEqual(deterministic_constraint_3.get_inequality_sign(), RelationalOperators.EQUALS)

        def test_inequality_sign_immutability(self):
            deterministic_constraint = DeterministicConstraint()
            deterministic_constraint.set_inequality_sign('<')

            with self.assertRaises(Exception):
                deterministic_constraint.set_inequality_sign('<')

            with self.assertRaises(Exception):
                deterministic_constraint.set_inequality_sign('>')

        def test_setting_sum_limit_digit_by_digit(self):
            deterministic_constraint = DeterministicConstraint()

            deterministic_constraint.add_character_to_sum_limit('-')
            self.assertFalse(deterministic_constraint.is_sum_limit_set())

            deterministic_constraint.add_character_to_sum_limit('1')
            self.assertTrue(deterministic_constraint.is_sum_limit_set())
            self.assertAlmostEqual(deterministic_constraint.get_sum_limit(), -1.0)

            deterministic_constraint.add_character_to_sum_limit('0')
            self.assertTrue(deterministic_constraint.is_sum_limit_set())
            self.assertAlmostEqual(deterministic_constraint.get_sum_limit(), -10.0)

            deterministic_constraint.add_character_to_sum_limit('.')
            self.assertTrue(deterministic_constraint.is_sum_limit_set())
            self.assertAlmostEqual(deterministic_constraint.get_sum_limit(), -10.0)

            deterministic_constraint.add_character_to_sum_limit('2')
            self.assertTrue(deterministic_constraint.is_sum_limit_set())
            self.assertAlmostEqual(deterministic_constraint.get_sum_limit(), -10.2)

        def test_get_sum_limit_before_setting_it(self):
            deterministic_constraint = DeterministicConstraint()
            with self.assertRaises(Exception):
                deterministic_constraint.get_sum_limit()

        def test_get_inequality_sign_before_setting_it(self):
            deterministic_constraint = DeterministicConstraint()
            with self.assertRaises(Exception):
                deterministic_constraint.get_inequality_sign()

        def test_invalid_inequality_sign(self):
            deterministic_constraint = DeterministicConstraint()
            with self.assertRaises(Exception):
                deterministic_constraint.set_inequality_sign('g')

        def test_set_and_get_attribute_name(self):
            deterministic_constraint = DeterministicConstraint()
            deterministic_constraint.set_attribute_name('attr')
            self.assertEqual(deterministic_constraint.get_attribute_name(), 'attr')

        def test_setting_attribute_name_by_character(self):
            deterministic_constraint = DeterministicConstraint()

            deterministic_constraint.add_character_to_attribute_name('a')
            self.assertEqual(deterministic_constraint.get_attribute_name(), 'a')

            deterministic_constraint.add_character_to_attribute_name('t')
            self.assertEqual(deterministic_constraint.get_attribute_name(), 'at')

            deterministic_constraint.add_character_to_attribute_name('t')
            self.assertEqual(deterministic_constraint.get_attribute_name(), 'att')

            deterministic_constraint.add_character_to_attribute_name('r')
            self.assertEqual(deterministic_constraint.get_attribute_name(), 'attr')

        def main(self):
            self.test_initial_conditions()
            self.test_sum_limit_consistency()
            self.test_inequality_sign_consistency()
            self.test_inequality_sign_immutability()
            self.test_setting_sum_limit_digit_by_digit()
            self.test_get_sum_limit_before_setting_it()
            self.test_get_inequality_sign_before_setting_it()
            self.test_invalid_inequality_sign()
            self.test_set_and_get_attribute_name()