import unittest
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from Utils.TailType import TailType


class CVaRConstraintUnitTest(unittest.TestCase):
    
    def test_initial_conditions(self):
        cvar_constraint = CVaRConstraint()
        self.assertTrue(cvar_constraint.is_risk_constraint())
        self.assertTrue(cvar_constraint.is_cvar_constraint())
        self.assertFalse(cvar_constraint.is_sum_limit_set())
        self.assertFalse(cvar_constraint.is_inequality_sign_set())
        self.assertFalse(cvar_constraint.is_percentage_of_scenarios_set())
        self.assertFalse(cvar_constraint.is_tail_type_set())
    
    def test_initial_conditions_from_expected_sum_constraint(self):
        expected_sum_constraint = ExpectedSumConstraint()
        expected_sum_constraint .set_attribute_name('attr')
        expected_sum_constraint.set_inequality_sign('>')
        expected_sum_constraint.set_sum_limit(-50)
        cvar_constraint = CVaRConstraint()
        cvar_constraint.initialize_from_expected_sum_constraint(
            expected_sum_constraint)
        self.assertEqual(cvar_constraint.get_attribute_name(),
                         expected_sum_constraint.get_attribute_name())
        self.assertEqual(cvar_constraint.get_inequality_sign(),
                         expected_sum_constraint.get_inequality_sign())
        self.assertEqual(cvar_constraint.get_sum_limit(),
                         expected_sum_constraint.get_sum_limit())
        self.assertFalse(cvar_constraint.is_percentage_of_scenarios_set())
        self.assertFalse(cvar_constraint.is_tail_type_set())
    
    def test_percentage_of_scenarios_consistency(self):
        cvar_constraint = CVaRConstraint()
        cvar_constraint.set_percentage_of_scenarios(5)
        self.assertEqual(cvar_constraint.get_percentage_of_scenarios(), 5)

        cvar_constraint = CVaRConstraint()
        cvar_constraint.set_percentage_of_scenarios(1)
        self.assertEqual(cvar_constraint.get_percentage_of_scenarios(), 1)

        cvar_constraint = CVaRConstraint()
        cvar_constraint.set_percentage_of_scenarios(100)
        self.assertEqual(cvar_constraint.get_percentage_of_scenarios(), 100)

        with self.assertRaises(Exception):
            cvar_constraint.set_percentage_of_scenarios(0)
        
        with self.assertRaises(Exception):
            cvar_constraint.set_percentage_of_scenarios(-5)

        with self.assertRaises(Exception):
            cvar_constraint.set_percentage_of_scenarios(105)
    
    def test_setting_percentage_of_scenarios_digit_by_digit(self):
        cvar_constraint = CVaRConstraint()

        cvar_constraint.add_character_to_percentage_of_scenarios('9')
        cvar_constraint.add_character_to_percentage_of_scenarios('.')
        cvar_constraint.add_character_to_percentage_of_scenarios('5')
        self.assertTrue(cvar_constraint.is_percentage_of_scenarios_set())
        self.assertAlmostEqual(cvar_constraint.get_percentage_of_scenarios(), 9.5)

        with self.assertRaises(Exception):
            cvar_constraint = CVaRConstraint()
            cvar_constraint.add_character_to_percentage_of_scenarios('-')
            cvar_constraint.add_character_to_percentage_of_scenarios('5')

        with self.assertRaises(Exception):
            cvar_constraint = CVaRConstraint()
            cvar_constraint.add_character_to_percentage_of_scenarios('1')
            cvar_constraint.add_character_to_percentage_of_scenarios('1')
            cvar_constraint.add_character_to_percentage_of_scenarios('1')
    
    def test_get_percentage_of_scenarios_before_setting_it(self):
        cvar_constraint = CVaRConstraint()
        with self.assertRaises(Exception):
            cvar_constraint.get_percentage_of_scenarios()
    
    def test_tail_type_consistency(self):
        cvar_constraint = CVaRConstraint()
        cvar_constraint.set_tail_type('h')
        self.assertEqual(cvar_constraint.get_tail_type(), TailType.HIGHEST)

        cvar_constraint = CVaRConstraint()
        cvar_constraint.set_tail_type('l')
        self.assertEqual(cvar_constraint.get_tail_type(), TailType.LOWEST)

        with self.assertRaises(Exception):
            cvar_constraint = CVaRConstraint()
            cvar_constraint.set_tail_type('a')
        
    def test_get_tail_type_before_setting_it(self):
        cvar_constraint = CVaRConstraint()
        with self.assertRaises(Exception):
            cvar_constraint.get_tail_type()


    def main(self):
        self.test_initial_conditions()
        self.test_initial_conditions_from_expected_sum_constraint()
        self.test_percentage_of_scenarios_consistency()
        self.test_setting_percentage_of_scenarios_digit_by_digit()
        self.test_get_percentage_of_scenarios_before_setting_it()
        self.test_get_tail_type_before_setting_it()

