from Utils.Heap import Heap
import unittest


class HeapUnitTest(unittest.TestCase):
    
    def test_min_heap(self):
        min_heap = Heap(is_max_heap=False)
        min_heap.push(4)
        min_heap.push(3)
        min_heap.push(6)
        self.assertEqual(min_heap.pop(), 3)
        self.assertEqual(min_heap.size(), 2)
        self.assertEqual(min_heap.sum(), 10)
        min_heap.push(5)
        self.assertEqual(min_heap.sum(), 15)
        self.assertEqual(min_heap.size(), 3)
        self.assertEqual(min_heap.pop(), 4)
        self.assertEqual(min_heap.sum(), 11)
        self.assertEqual(min_heap.size(), 2)
        self.assertEqual(min_heap.pop(), 5)
        self.assertEqual(min_heap.sum(), 6)
        self.assertEqual(min_heap.size(), 1)
        self.assertEqual(min_heap.pop(), 6)
        self.assertEqual(min_heap.size(), 0)
        self.assertEqual(min_heap.sum(), 0)

    def test_max_heap(self):
        max_heap = Heap(is_max_heap=True)
        max_heap.push(4)
        max_heap.push(3)
        max_heap.push(6)
        self.assertEqual(max_heap.sum(), 13)
        self.assertEqual(max_heap.pop(), 6)
        self.assertEqual(max_heap.sum(), 7)
        max_heap.push(5)
        self.assertEqual(max_heap.sum(), 12)
        self.assertEqual(max_heap.pop(), 5)
        self.assertEqual(max_heap.sum(), 7)
        self.assertEqual(max_heap.pop(), 4)
        self.assertEqual(max_heap.sum(), 3)
        self.assertEqual(max_heap.pop(), 3)
        self.assertEqual(max_heap.sum(), 0)


    def main(self):
        self.test_min_heap()
        self.test_max_heap()
