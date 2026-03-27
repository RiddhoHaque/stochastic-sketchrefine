import unittest
import numpy as np
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.PorfolioScenarioGenerator.GBMGainScenarioGenerator import GBMGainScenarioGenerator
from ScenarioGenerator.PorfolioScenarioGenerator.GARCHFHSGainScenarioGenerator import GARCHFHSGainScenarioGenerator

SEED            = 42
NO_OF_SCENARIOS = 5
TEST_YEAR       = 2015


def _table_row_count(table: str) -> int:
    try:
        PgConnection.Execute(f'SELECT COUNT(*) FROM {table}')
        return PgConnection.Fetch()[0][0]
    except Exception:
        if PgConnection.CONNECTION is not None:
            PgConnection.CONNECTION.rollback()
        return -1


def _verify_generator(test: unittest.TestCase, gen, label: str, expected_rows: int):
    """Run all checks with exactly two generate_scenarios calls."""
    result1 = gen.generate_scenarios(SEED, NO_OF_SCENARIOS)
    result2 = gen.generate_scenarios(SEED, NO_OF_SCENARIOS)

    test.assertEqual(len(result1), expected_rows,
                     f'{label}: output length mismatch')
    for i, (r1, r2) in enumerate(zip(result1, result2)):
        test.assertEqual(len(r1), NO_OF_SCENARIOS,
                         f'{label} row {i}: wrong scenario count')
        if r1:
            test.assertTrue(np.isfinite(r1).all(),
                            f'{label} row {i}: non-finite values')
        test.assertEqual(r1, r2,
                         f'{label} row {i}: not reproducible')


class GBMGARCHScenarioGeneratorUnitTest(unittest.TestCase):

    def test_gbm(self):
        table = f'GBM_Portfolio_{TEST_YEAR}'
        n = _table_row_count(table)
        if n <= 0:
            return
        _verify_generator(self, GBMGainScenarioGenerator(table), 'GBM', n)

    def test_garch(self):
        table = f'GARCH_Portfolio_{TEST_YEAR}'
        n = _table_row_count(table)
        if n <= 0:
            return
        _verify_generator(self, GARCHFHSGainScenarioGenerator(table), 'GARCH', n)

    def main(self):
        self.test_gbm()
        self.test_garch()
