import unittest
from OfflinePreprocessing.MeanAbsoluteDistance import MeanAbsoluteDistance


class MeanAbsoluteDistanceUnitTest(unittest.TestCase):

    def test_mad_finding(self):

        scenario1 = [0.1, 0.5, -1]
        scenario2 = [0.1, 1.5, -1.2]

        self.assertAlmostEqual(
            MeanAbsoluteDistance.get_distance(
                scenarios_1=scenario1,
                scenarios_2=scenario2), 0.4)
        
        with self.assertRaises(Exception):
            scenario1.append(1)
            MeanAbsoluteDistance.get_distance(
                scenarios_1=scenario1,
                scenarios_2=scenario2
            )

    def main(self):
        self.test_mad_finding()
