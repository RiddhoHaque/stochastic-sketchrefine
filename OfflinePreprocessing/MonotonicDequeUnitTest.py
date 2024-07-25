import unittest
from OfflinePreprocessing.MonotonicDeque import MonotonicDeque


class MonotonicDequeUnitTest(unittest.TestCase):

    def test_deque_monotonicity(self):
        deque = MonotonicDeque()
        deque.add_node(6, 0)
        deque.add_node(5, 1)
        deque.add_node(4, 1)
        tail_node = deque.get_tail(6)
        self.assertEqual(tail_node.get_index(), 6)
        self.assertEqual(tail_node.get_value(), 0)
        tail_node = deque.get_tail(5)
        self.assertEqual(tail_node.get_index(), 4)
        self.assertEqual(tail_node.get_value(), 1)
        deque.add_node(3, 2)
        tail_node = deque.get_tail(5)
        self.assertEqual(tail_node.get_index(), 4)
        self.assertEqual(tail_node.get_value(), 1)
        tail_node = deque.get_tail(3)
        self.assertEqual(tail_node.get_index(), 3)
        self.assertEqual(tail_node.get_value(), 2)
        deque.add_node(2, 2)
        tail_node = deque.get_tail(3)
        self.assertEqual(tail_node.get_index(), 2)
        self.assertEqual(tail_node.get_value(), 2)
        deque.add_node(1, 3)
        tail_node = deque.get_tail(2)
        self.assertEqual(tail_node.get_index(), 2)
        self.assertEqual(tail_node.get_value(), 2)
        deque.add_node(0, 4)
        tail_node = deque.get_tail(1)
        self.assertEqual(tail_node.get_index(), 1)
        self.assertEqual(tail_node.get_value(), 3)
        tail_node = deque.get_tail(0)
        self.assertEqual(tail_node.get_index(), 0)
        self.assertEqual(tail_node.get_value(), 4)



    def main(self):
        self.test_deque_monotonicity()
