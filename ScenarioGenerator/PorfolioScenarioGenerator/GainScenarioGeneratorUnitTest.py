import unittest
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.PorfolioScenarioGenerator.GainScenarioGenerator import GainScenarioGenerator


class GainScenarioGeneratorUnitTest(unittest.TestCase):
    def get_number_of_tuples(self):
        sql_query = """
        SELECT COUNT(*) FROM STOCK_INVESTMENTS_VOLATILITY_1X
        WHERE ticker='GOOG' OR ticker='AAPL'    
        """
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()[0][0]
    
    def test_reproducibility(self):
        gain_generator = GainScenarioGenerator(
        'Stock_Investments_Volatility_1x',
        "ticker='GOOG' or ticker='AAPL'")
        gains_1 = gain_generator.generate_scenarios(12330, 5)
        gains_2 = gain_generator.generate_scenarios(12330, 5)
        number_of_tuples = int(self.get_number_of_tuples())
        self.assertEqual(len(gains_1), number_of_tuples)
        self.assertEqual(len(gains_2), number_of_tuples)
        for i in range(len(gains_1)):
            
            self.assertEqual(len(gains_1[i]), 5)
            self.assertEqual(len(gains_2[i]), 5)
            
            for j in range(5):
                self.assertEqual(gains_1[i][j],
                                 gains_2[i][j])

    def main(self):
        self.test_reproducibility()



