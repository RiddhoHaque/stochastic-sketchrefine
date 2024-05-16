import unittest
from StochasticPackageQuery.Constraints.Constraint import Constraint
from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraint import DeterministicConstraint
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from StochasticPackageQuery.Objective.Objective import Objective
from StochasticPackageQuery.Query import Query
from Utils.ObjectiveType import ObjectiveType
from Utils.RelationalOperators import RelationalOperators
from Utils.Stochasticity import Stochasticity
from Utils.TailType import TailType


class QueryUnitTest(unittest.TestCase):

    def test_projected_attribute_consistency(self):
        query = Query()
        self.assertEqual(query.get_projected_attributes(), '')
        query.set_projected_attributes('attr1, attr2')
        self.assertEqual(query.get_projected_attributes(), 'attr1, attr2')

    def test_setting_projected_attribute_character_by_character(self):
        query = Query()
        query.add_character_to_projected_attributes('a')
        self.assertEqual(query.get_projected_attributes(), 'a')

        query.add_character_to_projected_attributes('t')
        self.assertEqual(query.get_projected_attributes(), 'at')

        query.add_character_to_projected_attributes('t')
        self.assertEqual(query.get_projected_attributes(), 'att')

        query.add_character_to_projected_attributes('r')
        self.assertEqual(query.get_projected_attributes(), 'attr')

        query.add_character_to_projected_attributes('1')
        self.assertEqual(query.get_projected_attributes(), 'attr1')

        query.add_character_to_projected_attributes(',')
        self.assertEqual(query.get_projected_attributes(), 'attr1,')

        query.add_character_to_projected_attributes(' ')
        self.assertEqual(query.get_projected_attributes(), 'attr1, ')

        query.add_character_to_projected_attributes('a')
        self.assertEqual(query.get_projected_attributes(), 'attr1, a')

        query.add_character_to_projected_attributes('t')
        self.assertEqual(query.get_projected_attributes(), 'attr1, at')

        query.add_character_to_projected_attributes('t')
        self.assertEqual(query.get_projected_attributes(), 'attr1, att')

        query.add_character_to_projected_attributes('r')
        self.assertEqual(query.get_projected_attributes(), 'attr1, attr')

        query.add_character_to_projected_attributes('2')
        self.assertEqual(query.get_projected_attributes(), 'attr1, attr2')

    def test_package_alias_consistency(self):
        query = Query()
        self.assertEqual(query.get_package_alias(), '')
        query.set_package_alias('portfolio')
        self.assertEqual(query.get_package_alias(), 'portfolio')

    def test_set_package_alias_character_by_character(self):
        query = Query()
        self.assertEqual(query.get_package_alias(), '')
        query.add_character_to_package_alias('p')
        self.assertEqual(query.get_package_alias(), 'p')
        query.add_character_to_package_alias('1')
        self.assertEqual(query.get_package_alias(), 'p1')

    def test_relation_consistency(self):
        query = Query()
        self.assertEqual(query.get_relation(), '')
        query.set_relation('t1')
        self.assertEqual(query.get_relation(), 't1')

    def test_set_relation_character_by_character(self):
        query = Query()
        query.add_character_to_relation('t')
        self.assertEqual(query.get_relation(), 't')
        query.add_character_to_relation('1')
        self.assertEqual(query.get_relation(), 't1')

    def test_relation_alias_consistency(self):
        query = Query()
        self.assertEqual(query.get_relation_alias(), '')
        query.set_relation_alias('t')
        self.assertEqual(query.get_relation_alias(), 't')

    def test_set_relation_alias_character_by_character(self):
        query = Query()
        query.add_character_to_relation_alias('t')
        self.assertEqual(query.get_relation_alias(), 't')
        query.add_character_to_relation_alias('r')
        self.assertEqual(query.get_relation_alias(), 'tr')

    def test_base_predicate_consistency(self):
        query = Query()
        self.assertEqual(query.get_base_predicate(), '')
        query.set_base_predicate("where status='free'")
        self.assertEqual(query.get_base_predicate(), "where status='free'")

    def test_set_base_predicate_character_by_character(self):
        query = Query()
        base_predicate = "where status='free'"
        for character in base_predicate:
            query.add_character_to_base_predicate(character)
        self.assertEqual(query.get_base_predicate(), base_predicate)

    def test_add_constraint(self):
        query = Query()
        constraint = Constraint()
        query.add_constraint(constraint)
        self.assertEqual(len(query.get_constraints()), 1)
        self.assertEqual(query.get_constraints()[0], constraint)

    def test_add_repeat_constraint(self):
        query = Query()
        query.add_repeat_constraint()
        self.assertEqual(len(query.get_constraints()), 1)
        self.assertTrue(query.get_constraints()[0].is_repeat_constraint())

    def test_add_digit_to_repeat_constraint(self):
        query = Query()
        with self.assertRaises(Exception):
            query.add_digit_to_repeat_constraint(1)
        constraint = Constraint()
        query.add_constraint(constraint)
        with self.assertRaises(Exception):
            query.add_digit_to_repeat_constraint(1)
        query.add_repeat_constraint()
        query.add_digit_to_repeat_constraint(1)
        query.add_digit_to_repeat_constraint(0)
        self.assertEqual(query.get_constraints()[-1].get_repetition_limit(), 10)

    def test_add_package_size_constraint(self):
        query = Query()
        with self.assertRaises(Exception):
            query.add_digit_to_package_size_constraint(1)
        constraint = Constraint()
        query.add_constraint(constraint)
        with self.assertRaises(Exception):
            query.add_digit_to_package_size_constraint(1)
        query.add_package_size_constraint()
        query.add_digit_to_package_size_constraint(1)
        query.add_digit_to_package_size_constraint(0)
        self.assertEqual(query.get_constraints()[-1].get_package_size_limit(), 10)

    def test_add_character_to_deterministic_constraint_attribute(self):
        query = Query()
        with self.assertRaises(Exception):
            query.add_character_to_attribute_name('a')
        constraint = Constraint()
        query.add_constraint(constraint)
        with self.assertRaises(Exception):
            query.add_character_to_attribute_name('a')
        query.add_deterministic_constraint()
        query.add_character_to_attribute_name('a')
        query.add_character_to_attribute_name('t')
        query.add_character_to_attribute_name('t')
        query.add_character_to_attribute_name('r')
        self.assertEqual(query.get_constraints()[-1].get_attribute_name(), 'attr')

    def test_add_character_to_expected_sum_constraint_attribute(self):
        query = Query()
        query.add_expected_sum_constraint()
        query.add_character_to_attribute_name('a')
        query.add_character_to_attribute_name('t')
        query.add_character_to_attribute_name('t')
        query.add_character_to_attribute_name('r')
        self.assertEqual(query.get_constraints()[-1].get_attribute_name(), 'attr')

    def test_add_character_to_var_constraint_attribute(self):
        query = Query()
        query.add_var_constraint()
        query.add_character_to_attribute_name('a')
        query.add_character_to_attribute_name('t')
        query.add_character_to_attribute_name('t')
        query.add_character_to_attribute_name('r')
        self.assertEqual(query.get_constraints()[-1].get_attribute_name(), 'attr')

    def test_deterministic_constraint_inequality(self):
        query = Query()
        with self.assertRaises(Exception):
            query.set_constraint_inequality_sign('>')
        constraint = Constraint()
        query.add_constraint(constraint)
        with self.assertRaises(Exception):
            query.set_constraint_inequality_sign('<')
        query.add_deterministic_constraint()
        query.set_constraint_inequality_sign('<')
        self.assertEqual(query.get_constraints()[-1].get_inequality_sign(),
                         RelationalOperators.LESS_THAN_OR_EQUAL_TO)

    def test_expected_sum_constraint_inequality(self):
        query = Query()
        query.add_expected_sum_constraint()
        query.set_constraint_inequality_sign('>')
        self.assertEqual(query.get_constraints()[-1].get_inequality_sign(),
                         RelationalOperators.GREATER_THAN_OR_EQUAL_TO)

    def test_var_constraint_inequality(self):
        query = Query()
        query.add_var_constraint()
        query.set_constraint_inequality_sign('=')
        self.assertEqual(query.get_constraints()[-1].get_inequality_sign(),
                         RelationalOperators.EQUALS)
    
    def test_set_constraint_sum_limit(self):
        query = Query()
        with self.assertRaises(Exception):
            query.add_character_to_constraint_sum_limit('0')
        constraint = Constraint()
        query.add_constraint(constraint)
        with self.assertRaises(Exception):
            query.add_character_to_constraint_sum_limit('0')
        query.add_deterministic_constraint()
        query.add_character_to_constraint_sum_limit('-')
        query.add_character_to_constraint_sum_limit('1')
        query.add_character_to_constraint_sum_limit('0')
        query.add_character_to_constraint_sum_limit('.')
        query.add_character_to_constraint_sum_limit('5')
        self.assertAlmostEqual(query.get_constraints()[-1].get_sum_limit(), -10.5)

    def test_convert_deterministic_constraint_to_var_constraint(self):
        query = Query()
        with self.assertRaises(Exception):
            query.convert_final_deterministic_constraint_to_var_constraint()
        query.add_constraint(Constraint())
        with self.assertRaises(Exception):
            query.convert_final_deterministic_constraint_to_var_constraint()
        deterministic_constraint = DeterministicConstraint()
        deterministic_constraint.set_attribute_name('A')
        deterministic_constraint.set_inequality_sign('>')
        deterministic_constraint.set_sum_limit(10)
        query.add_constraint(deterministic_constraint)
        query.convert_final_deterministic_constraint_to_var_constraint()
        last_constraint = query.get_constraints()[-1]
        self.assertTrue(last_constraint.is_var_constraint())
        self.assertFalse(last_constraint.is_deterministic_constraint())
        self.assertEqual(last_constraint.get_attribute_name(),
                         deterministic_constraint.get_attribute_name())
        self.assertEqual(last_constraint.get_inequality_sign(),
                         deterministic_constraint.get_inequality_sign())
        self.assertEqual(last_constraint.get_sum_limit(),
                         deterministic_constraint.get_sum_limit())
    
    def test_convert_expected_sum_constraint_to_cvar_constraint(self):
        query = Query()
        with self.assertRaises(Exception):
            query.convert_final_expected_sum_constraint_to_cvar_constraint(self)
        query.add_constraint(Constraint())
        with self.assertRaises(Exception):
            query.convert_final_expected_sum_constraint_to_cvar_constraint(self)
        expected_sum_constraint = ExpectedSumConstraint()
        expected_sum_constraint.set_attribute_name('A')
        expected_sum_constraint.set_inequality_sign('>')
        expected_sum_constraint.set_sum_limit(10)
        query.add_constraint(expected_sum_constraint)
        query.convert_final_expected_sum_constraint_to_cvar_constraint()
        last_constraint = query.get_constraints()[-1]
        self.assertTrue(last_constraint.is_cvar_constraint())
        self.assertFalse(last_constraint.is_expected_sum_constraint())
        self.assertEqual(last_constraint.get_attribute_name(),
                         expected_sum_constraint.get_attribute_name())
        self.assertEqual(last_constraint.get_inequality_sign(),
                         expected_sum_constraint.get_inequality_sign())
        self.assertEqual(last_constraint.get_sum_limit(),
                         expected_sum_constraint.get_sum_limit())

    def test_set_constraint_probability_threshold(self):
        query = Query()
        with self.assertRaises(Exception):
            query.add_character_to_constraint_probability_threshold('1')
        query.add_constraint(Constraint())
        with self.assertRaises(Exception):
            query.add_character_to_constraint_probability_threshold('1')
        query.add_var_constraint()
        query.add_character_to_constraint_probability_threshold('0')
        query.add_character_to_constraint_probability_threshold('.')
        query.add_character_to_constraint_probability_threshold('9')
        query.add_character_to_constraint_probability_threshold('5')
        self.assertAlmostEqual(query.get_constraints()[-1].get_probability_threshold(), 0.95)
    
    def test_set_constraint_tail_type(self):
        query = Query()
        with self.assertRaises(Exception):
            query.add_character_to_constraint_tail_type('l')
        query.add_constraint(Constraint())
        with self.assertRaises(Exception):
            query.add_character_to_constraint_tail_type('l')
        query.add_cvar_constraint()
        query.add_character_to_constraint_tail_type('l')
        self.assertEqual(query.get_constraints()[-1].get_tail_type(),
                         TailType.LOWEST)
    
    def test_set_constraint_percentage_of_scenarios(self):
        query = Query()
        with self.assertRaises(Exception):
            query.add_character_to_constraint_percentage_of_scenarios('9')
        query.add_constraint(Constraint())
        with self.assertRaises(Exception):
            query.add_character_to_constraint_percentage_of_scenarios('9')
        query.add_cvar_constraint()
        query.add_character_to_constraint_percentage_of_scenarios('9')
        self.assertAlmostEqual(query.get_constraints()[-1].get_percentage_of_scenarios(), 9)

    def test_objective_consistency(self):
        query = Query()
        objective = Objective()
        query.set_objective(objective)
        self.assertEqual(query.get_objective(), objective)
        query.set_objective_type(is_maximization=True)
        self.assertEqual(query.get_objective().get_objective_type(), ObjectiveType.MAXIMIZATION)
        query.set_objective_type(is_maximization=False)
        self.assertEqual(query.get_objective().get_objective_type(), ObjectiveType.MINIMIZATION)
        query.set_objective_stochasticity(is_stochastic=True)
        self.assertEqual(query.get_objective().get_stochasticity(), Stochasticity.STOCHASTIC)
        query.set_objective_stochasticity(is_stochastic=False)
        self.assertEqual(query.get_objective().get_stochasticity(), Stochasticity.DETERMINISTIC)
        query.add_character_to_objective_attribute('a')
        query.add_character_to_objective_attribute('t')
        query.add_character_to_objective_attribute('t')
        query.add_character_to_objective_attribute('r')
        self.assertEqual(query.get_objective().get_attribute_name(), 'attr')

    def main(self):
        self.test_projected_attribute_consistency()
        self.test_setting_projected_attribute_character_by_character()
        self.test_package_alias_consistency()
        self.test_set_package_alias_character_by_character()
        self.test_relation_consistency()
        self.test_set_relation_character_by_character()
        self.test_relation_alias_consistency()
        self.test_set_relation_alias_character_by_character()
        self.test_base_predicate_consistency()
        self.test_set_base_predicate_character_by_character()
        self.test_add_constraint()
        self.test_add_repeat_constraint()
        self.test_add_digit_to_repeat_constraint()
        self.test_add_package_size_constraint()
        self.test_add_character_to_deterministic_constraint_attribute()
        self.test_add_character_to_expected_sum_constraint_attribute()
        self.test_add_character_to_var_constraint_attribute()
        self.test_deterministic_constraint_inequality()
        self.test_expected_sum_constraint_inequality()
        self.test_var_constraint_inequality()
        self.test_set_constraint_sum_limit()
        self.test_convert_deterministic_constraint_to_var_constraint()
        self.test_convert_expected_sum_constraint_to_cvar_constraint()
        self.test_set_constraint_probability_threshold()
        self.test_set_constraint_tail_type()
        self.test_set_constraint_percentage_of_scenarios()
        self.test_objective_consistency()