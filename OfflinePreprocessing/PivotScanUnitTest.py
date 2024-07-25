import unittest
from OfflinePreprocessing.PivotScan import PivotScan
from PgConnection.PgConnection import PgConnection
from DbInfo.PortfolioInfo import PortfolioInfo


class PivotScanUnitTest(unittest.TestCase):

    def test_pivot_scan_for_aapl_stocks(self):
        relation = 'Stock_Investments_90'
        sql_query = 'SELECT id FROM ' + relation\
            + " WHERE ticker>='Z' ORDER BY id"
        PgConnection.Execute(sql_query)
        tuples = PgConnection.Fetch()
        ids = []
        for id in tuples:
            ids.append(id[0])
        distances_and_ids = \
            PivotScan.get_ids_with_increasing_distances(
                ids, pivots=[0], attribute='gain',
                relation=relation, dbinfo=PortfolioInfo,
                init_seed=123434
            )
        last_distance = -1
        for idx in range(len(distances_and_ids)):
            distance, id = distances_and_ids[idx]
            self.assertTrue(distance >= last_distance)
            self.assertTrue(distance >= 0)

    def main(self):
        self.test_pivot_scan_for_aapl_stocks()
