import unittest
from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Parser.Transition.Transition import Transition
from StochasticPackageQuery.Query import Query


class StateUnitTest(unittest.TestCase):

    def test_process_keeps_query_unchanged(self):
        state = State()
        query = Query()
        self.assertEqual(state.process(query, 'a'), query)
        
    def test_transitions_added_correctly(self):
        state = State()
        next_state = State()
        transition1 = Transition('a', next_state)
        transition2 = Transition('b', next_state)
        state.add_transition(transition1)
        state.add_transition(transition2)
        with self.assertRaises(Exception):
            state.add_transition(Transition('a', next_state))
        self.assertEqual(len(state.get_transitions()), 2)
        self.assertEqual(state.get_transitions()[0], transition1)
        self.assertEqual(state.get_transitions()[1], transition2)

    def test_get_next_state_correctly(self):
        state = State()
        next_state1 = State()
        next_state2 = State()
        transition1 = Transition('a', next_state1)
        transition2 = Transition('b', next_state2)
        state.add_transition(transition1)
        state.add_transition(transition2)
        self.assertEqual(state.get_next_state('a'), next_state1)
        self.assertEqual(state.get_next_state('b'), next_state2)
        with self.assertRaises(Exception):
            state.get_next_state('c')


    def main(self):
        self.test_process_keeps_query_unchanged()
        self.test_transitions_added_correctly()
        self.test_get_next_state_correctly()
        
