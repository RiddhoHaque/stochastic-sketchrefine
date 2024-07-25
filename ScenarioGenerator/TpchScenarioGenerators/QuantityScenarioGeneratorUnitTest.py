from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.TpchScenarioGenerators.QuantityScenarioGenerator import QuantityScenarioGenerator
import unittest
import numpy as np


class QuantityScenarioGeneratorUnitTest(unittest.TestCase):

    def create_mock_table(self):
        PgConnection.Execute(
            'DROP TABLE IF EXISTS MOCK_QUANTITY_TABLE;'
        )
        PgConnection.Execute(
            """
            CREATE TABLE MOCK_QUANTITY_TABLE(
                id int not null unique,
                quantity float,
                quantity_mean float,
                quantity_variance float,
                quantity_variance_coeff float
            );
            """
        )

    def populate_mock_table(self):
        PgConnection.Execute(
            """
            INSERT INTO MOCK_QUANTITY_TABLE VALUES(1, 5, -0.1, 1, 1);
            INSERT INTO MOCK_QUANTITY_TABLE VALUES(2, 7, 0.0, 2, 20);
            """
        )
    
    def cleanup_mock_table(self):
        PgConnection.Execute(
            """
            DROP TABLE IF EXISTS MOCK_QUANTITY_TABLE;
            """
        )
    
    def is_almost_equal(self, v1: float, v2: float) -> bool:
        if np.abs(v1-v2)/v2 >= 0.01:
            print(v1, 'and', v2, 'are not close enough')
        return np.abs(v1 - v2)/v2 < 0.01
    
    def test_quantity_scenario_generator(self):
        quantity_generator = QuantityScenarioGenerator(
            relation='MOCK_QUANTITY_TABLE',
            base_predicate='ID>=1 and ID<=2'
        )
        no_of_scenarios = 50000
        quantities = quantity_generator.generate_scenarios(
            seed = 122446124, no_of_scenarios=no_of_scenarios
        )
        self.assertEqual(len(quantities), 2)
        self.assertEqual(len(quantities[0]), no_of_scenarios)
        self.assertEqual(len(quantities[0]), no_of_scenarios)
        self.assertTrue(self.is_almost_equal(np.average(quantities[0]), 4.9))
        self.assertTrue(self.is_almost_equal(np.std(quantities[0]), np.sqrt(1*1)))
        self.assertTrue(self.is_almost_equal(np.average(quantities[1]), 7))
        self.assertTrue(self.is_almost_equal(np.std(quantities[1]), np.sqrt(2*20)))
    
    def test_vertical_bulk_generation(self):
        quantity_generator = QuantityScenarioGenerator(
            relation='MOCK_QUANTITY_TABLE',
        )
        quantities = quantity_generator.generate_scenarios(
            seed = 1132431359, no_of_scenarios=1
        )
        self.assertEqual(len(quantities), 2)
        self.assertEqual(len(quantities[0]), 1)
        self.assertEqual(len(quantities[1]), 1)
    
    def main(self):
        self.create_mock_table()
        self.populate_mock_table()
        self.test_quantity_scenario_generator()
        self.test_vertical_bulk_generation()
        self.cleanup_mock_table()