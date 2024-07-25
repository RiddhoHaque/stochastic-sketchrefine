from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.TpchScenarioGenerators.PriceScenarioGenerator import PriceScenarioGenerator
import unittest
import numpy as np


class PriceScenarioGeneratorUnitTest(unittest.TestCase):

    def create_mock_table(self):
        PgConnection.Execute(
            'DROP TABLE IF EXISTS MOCK_PRICE_TABLE;'
        )
        PgConnection.Execute(
            """
            CREATE TABLE MOCK_PRICE_TABLE(
                id int not null unique,
                price float,
                price_mean float,
                price_variance float,
                price_variance_coeff float
            );
            """
        )
    
    def populate_mock_table(self):
        PgConnection.Execute(
            """
            INSERT INTO MOCK_PRICE_TABLE VALUES(1, 5, -0.1, 1, 1);
            INSERT INTO MOCK_PRICE_TABLE VALUES(2, 7, 0.0, 2, 20);
            """
        )
    
    def cleanup_mock_table(self):
        PgConnection.Execute(
            """
            DROP TABLE IF EXISTS MOCK_PRICE_TABLE;
            """
        )
    
    def is_almost_equal(self, v1: float, v2: float) -> bool:
        if np.abs(v1 - v2)/v2 >= 0.01:
            print(v1, 'and', v2, 'are not close enough')
        return np.abs(v1 - v2)/v2 < 0.01

    def test_price_scenario_generator(self):
        price_generator = PriceScenarioGenerator(
            relation='MOCK_PRICE_TABLE',
            base_predicate='ID >= 1 AND ID <= 2'
        )
        no_of_scenarios = 50000
        prices = price_generator.generate_scenarios(
            seed = 3785731134, no_of_scenarios=no_of_scenarios
        )
        self.assertEqual(len(prices), 2)
        self.assertEqual(len(prices[0]), no_of_scenarios)
        self.assertEqual(len(prices[1]), no_of_scenarios)
        self.assertTrue(self.is_almost_equal(np.average(prices[0]), 4.9))
        self.assertTrue(self.is_almost_equal(np.std(prices[0]), np.sqrt(1*1)))
        self.assertTrue(self.is_almost_equal(np.average(prices[1]), 7))
        self.assertTrue(self.is_almost_equal(np.std(prices[1]), np.sqrt(2*20)))
    
    def test_vertical_bulk_generation(self):
        price_generator = PriceScenarioGenerator(
            relation='MOCK_PRICE_TABLE',
        )
        prices = price_generator.generate_scenarios(
            seed = 10231, no_of_scenarios=1
        )
        self.assertEqual(len(prices), 2)
        self.assertEqual(len(prices[0]), 1)
        self.assertEqual(len(prices[1]), 1)
    
    def main(self):
        self.create_mock_table()
        self.populate_mock_table()
        self.test_price_scenario_generator()
        self.test_vertical_bulk_generation()
        self.cleanup_mock_table()
