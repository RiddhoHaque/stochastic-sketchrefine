from PgConnection.PgConnection import PgConnection
from ValueGenerator.ValueGenerator import ValueGenerator
import unittest


class ValueGeneratorUnitTest(unittest.TestCase):

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
    

    def test_value_generator(self):
        value_generator = ValueGenerator(
            relation='MOCK_PRICE_TABLE',
            attribute='price_mean',
            base_predicate='ID >= 1 AND ID <= 2'
        )
        data = value_generator.get_values()
        self.assertEqual(float(data[0][0]), -0.1)
        self.assertEqual(float(data[1][0]), 0.0)

    def main(self):
        self.create_mock_table()
        self.populate_mock_table()
        self.test_value_generator()
        self.cleanup_mock_table()
