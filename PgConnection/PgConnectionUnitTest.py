from PgConnection.PgConnection import PgConnection
import unittest


class PgConnectionUnitTest(unittest.TestCase):
    
    def create_mock_table(self):
        PgConnection.Execute(
            'DROP TABLE IF EXISTS Mock_Table;')
        PgConnection.Execute(
            """
            CREATE TABLE Mock_Table(
                id int not null unique,
                fname varchar(10),
                lname varchar(10)
            );
            """
        )
    
    def populate_mock_table(self):
        PgConnection.Execute(
            """
            INSERT INTO Mock_Table VALUES(1, 'abc', 'def');
            """
        )
        PgConnection.Execute(
            """
            INSERT INTO Mock_Table VALUES(2, 'ghi', 'jkl');
            """
        )
    
    def query_mock_table(self):
        PgConnection.Execute(
            """
            SELECT * FROM Mock_Table;
            """
        )
        return PgConnection.Fetch()
    
    def check_query_results(self, query_results):
        self.assertTrue(query_results[0], (1, 'abc', 'def'))
        self.assertTrue(query_results[1], (2, 'ghi', 'jkl'))

    def cleanup_mock_table(self):
        PgConnection.Execute(
            'DROP TABLE IF EXISTS Mock_Table;')

    def main(self):
        self.create_mock_table()
        self.populate_mock_table()
        self.check_query_results(self.query_mock_table())
        self.cleanup_mock_table()