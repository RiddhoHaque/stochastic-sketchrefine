import unittest
from StochasticPackageQuery.Constraints.PackageSizeConstraint.PackageSizeConstraint import PackageSizeConstraint
from Utils.RelationalOperators import RelationalOperators

class PackageSizeConstraintUnitTest(unittest.TestCase):

    def test_initial_conditions(self):
        package_size_constraint = PackageSizeConstraint()
        self.assertTrue(package_size_constraint.is_package_size_constraint())
        self.assertFalse(package_size_constraint.is_package_size_limit_set())
        self.assertFalse(package_size_constraint.is_inequality_sign_set())

    def test_package_size_limit_consistency(self):
        package_size_constraint = PackageSizeConstraint()
        package_size_constraint.set_package_size_limit(5)
        self.assertEqual(package_size_constraint.get_package_size_limit(), 5)

        package_size_constraint = PackageSizeConstraint()
        package_size_constraint.set_package_size_limit(10)
        self.assertEqual(package_size_constraint.get_package_size_limit(), 10)

        with self.assertRaises(ValueError):
            package_size_constraint.set_package_size_limit(-1)

    def test_inequality_sign_consistency(self):
        package_size_constraint_1 = PackageSizeConstraint()
        package_size_constraint_2 = PackageSizeConstraint()
        package_size_constraint_3 = PackageSizeConstraint()

        package_size_constraint_1.set_inequality_sign('<')
        self.assertEqual(package_size_constraint_1.get_inequality_sign(), RelationalOperators.LESS_THAN_OR_EQUAL_TO)

        package_size_constraint_2.set_inequality_sign('>')
        self.assertEqual(package_size_constraint_2.get_inequality_sign(), RelationalOperators.GREATER_THAN_OR_EQUAL_TO)

        package_size_constraint_3.set_inequality_sign('=')
        self.assertEqual(package_size_constraint_3.get_inequality_sign(), RelationalOperators.EQUALS)

    def test_inequality_size_immutability(self):
        package_size_constraint = PackageSizeConstraint()
        package_size_constraint.set_inequality_sign('<')

        with self.assertRaises(Exception):
            package_size_constraint.set_inequality_sign('<')

        with self.assertRaises(Exception):
            package_size_constraint.set_inequality_sign('>')
    def test_setting_package_size_limit_digit_by_digit(self):
        package_size_constraint = PackageSizeConstraint()

        package_size_constraint.add_digit_to_package_size_limit(1)
        self.assertTrue(package_size_constraint.is_package_size_limit_set())
        self.assertEqual(package_size_constraint.get_package_size_limit(), 1)

        package_size_constraint.add_digit_to_package_size_limit(0)
        self.assertTrue(package_size_constraint.is_package_size_limit_set())
        self.assertEqual(package_size_constraint.get_package_size_limit(), 10)

        package_size_constraint.add_digit_to_package_size_limit(2)
        self.assertTrue(package_size_constraint.is_package_size_limit_set())
        self.assertEqual(package_size_constraint.get_package_size_limit(), 102)

        with self.assertRaises(ValueError):
            package_size_constraint.add_digit_to_package_size_limit(-1)

        with self.assertRaises(ValueError):
            package_size_constraint.add_digit_to_package_size_limit(10)

    def test_get_package_size_limit_before_setting_it(self):
        package_size_constraint = PackageSizeConstraint()
        with self.assertRaises(Exception):
            package_size_constraint.get_package_size_limit()

    def test_get_inequality_sign_before_setting_it(self):
        package_size_constraint = PackageSizeConstraint()
        with self.assertRaises(Exception):
            package_size_constraint.get_inequality_sign()

    def test_invalid_inequality_sign(self):
        package_size_constraint = PackageSizeConstraint()
        with self.assertRaises(Exception):
            package_size_constraint.set_inequality_sign('g')

    def main(self):
        self.test_initial_conditions()
        self.test_package_size_limit_consistency()
        self.test_inequality_sign_consistency()
        self.test_inequality_size_immutability()
        self.test_setting_package_size_limit_digit_by_digit()
        self.test_get_package_size_limit_before_setting_it()
        self.test_get_inequality_sign_before_setting_it()
        self.test_invalid_inequality_sign()