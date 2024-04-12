import unittest
from Utils.ObjectiveType import ObjectiveType
from Utils.Stochasticity import Stochasticity
from StochasticPackageQuery.Objective.Objective import Objective


class ObjectiveUnitTest(unittest.TestCase):
    def test_initial_conditions(self):
        objective = Objective()
        self.assertFalse(objective.is_objective_type_set())
        self.assertFalse(objective.is_stochasticity_set())

        with self.assertRaises(Exception):
            objective.get_objective_type()

        with self.assertRaises(Exception):
            objective.get_stochasticity()

    def test_objective_type_consistency(self):
        objective = Objective()
        objective.set_objective_type(is_maximization=True)
        self.assertEqual(objective.get_objective_type(), ObjectiveType.MAXIMIZATION)
        self.assertTrue(objective.is_objective_type_set())

        objective = Objective()
        objective.set_objective_type(is_maximization=False)
        self.assertEqual(objective.get_objective_type(), ObjectiveType.MINIMIZATION)
        self.assertTrue(objective.is_objective_type_set())

    def test_stochasticity_consistency(self):
        objective = Objective()
        objective.set_stochasticity(is_stochastic=True)
        self.assertEqual(objective.get_stochasticity(), Stochasticity.STOCHASTIC)
        self.assertTrue(objective.is_stochasticity_set())

        objective = Objective()
        objective.set_stochasticity(is_stochastic=False)
        self.assertEqual(objective.get_stochasticity(), Stochasticity.DETERMINISTIC)
        self.assertTrue(objective.is_stochasticity_set())

    def test_attribute_name_consistency(self):
        objective = Objective()
        self.assertEqual(objective.get_attribute_name(), '')
        objective.set_attribute_name('attr')
        self.assertEqual(objective.get_attribute_name(), 'attr')

    def test_setting_attribute_name_character_by_character(self):
        objective = Objective()
        self.assertEqual(objective.get_attribute_name(), '')
        objective.add_character_to_attribute_name('a')
        self.assertEqual(objective.get_attribute_name(), 'a')
        objective.add_character_to_attribute_name('t')
        self.assertEqual(objective.get_attribute_name(), 'at')
        objective.add_character_to_attribute_name('t')
        self.assertEqual(objective.get_attribute_name(), 'att')
        objective.add_character_to_attribute_name('r')
        self.assertEqual(objective.get_attribute_name(), 'attr')

    def main(self):
        self.test_initial_conditions()
        self.test_objective_type_consistency()
        self.test_stochasticity_consistency()
        self.test_attribute_name_consistency()
        self.test_setting_attribute_name_character_by_character()
