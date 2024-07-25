import unittest
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator


class ScenarioGeneratorUnitTest(unittest.TestCase):

    def test_dimensions(self):
        scenario_generator = ScenarioGenerator(
            relation='Jane_Doe'
        )
        scenarios = \
            scenario_generator.generate_scenarios(
                seed = 122435, no_of_scenarios=100
            )
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(len(scenarios[0]), 100)
        for _ in range(100):
            self.assertEqual(scenarios[0][_], 0)
    
    def main(self):
        self.test_dimensions()