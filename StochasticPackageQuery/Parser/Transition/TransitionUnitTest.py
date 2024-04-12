import unittest
from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Parser.Transition.Transition import Transition


class TransitionUnitTest(unittest.TestCase):

    def test_firing(self):
        state = State()
        transition = Transition('a', state)
        self.assertTrue(transition.fires('a'))
        self.assertFalse(transition.fires('b'))
        self.assertEqual(transition.get_next_state(), state)

    def main(self):
        self.test_firing()
