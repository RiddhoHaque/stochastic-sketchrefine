from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class RubberDuckState(State):

    def process(self, query: Query, char: chr) -> Query:
        print('Quack quack, ', char)
        return query