import unittest
from OfflinePreprocessing.OptimalPartitioning import OptimalPartitioning


class OptimalPartitioningUnitTest(unittest.TestCase):

    def test_partitioning(self):
        distances = [(0, 1), (1.8, 3), (2.1, 2), (2.2, 6), (2.3, 4), (2.4, 5),
                     (4.8, 6), (4.9, 7), (6.9, 8), (7.0, 9), (7.1, 10)]
        next_division = OptimalPartitioning.get_next_divisions(distances, diameter=2)
        self.assertEqual(next_division[0], 1)
        for _ in range(1, 6):
            self.assertEqual(next_division[_], 6)
        for _ in range(6, 8):
            self.assertEqual(next_division[_], 8)
        for _ in range(8, 11):
            self.assertEqual(next_division[_], 11)

        distances = [(0, 1), (1.8, 3), (2.1, 2), (2.2, 6), (2.3, 4), (2.4, 5),
                     (4.8, 6), (4.9, 7), (6.9, 8), (7.0, 9), (7.1, 10)]
        next_division = OptimalPartitioning.get_next_divisions(distances, diameter=0.05)
        for _ in range(len(next_division)):
            self.assertEqual(next_division[_], _+1)

        distances = [(0, 1), (1.8, 3), (2.1, 2), (2.2, 6), (2.3, 4), (2.4, 5),
                     (4.8, 6), (4.9, 7), (6.9, 8), (7.0, 9), (7.1, 10)]
        next_division = OptimalPartitioning.get_next_divisions(distances, diameter=0.05)
        for _ in range(len(next_division)):
            self.assertEqual(next_division[_], _+1)

        print(next_division)

    def main(self):
        self.test_partitioning()
