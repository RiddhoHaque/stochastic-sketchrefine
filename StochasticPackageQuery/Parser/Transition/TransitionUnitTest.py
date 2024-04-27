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
        anything_but_transition = Transition('a', state,
                                             anything_but_transition=True)
        self.assertTrue(anything_but_transition.fires('b'))
        self.assertTrue(anything_but_transition.fires('c'))
        self.assertFalse(anything_but_transition.fires('a'))

    def test_accept_anything_transition(self):
        state = State()
        transition = Transition('a', state,
                                anything_but_transition=True,
                                accept_anything_transition= True)
        self.assertTrue(transition.fires('a'))
        self.assertTrue(transition.fires('b'))

    def main(self):
        self.test_firing()
        self.test_accept_anything_transition()
