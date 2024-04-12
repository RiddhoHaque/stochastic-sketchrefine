from StochasticPackageQuery.Query import Query
from StochasticPackageQuery.Parser.Transition.Transition import Transition


class State:

    def __init__(self) -> None:
        self.__transitions = []

    def process(self, query: Query, char: chr) -> Query:
        return query
    
    def get_transitions(self) -> [Transition]:
        return self.__transitions
    
    def add_transition(self, transition: Transition):
        for existing_transition in self.__transitions:
            if existing_transition.get_trigger() == transition.get_trigger():
                raise Exception
        self.__transitions.append(transition)
    
    def get_next_state(self, char: chr):
        for transition in self.__transitions:
            if transition.fires(char):
                return transition.get_next_state()
        raise Exception
