from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class AddDeterministicConstraintState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_deterministic_constraint()
        return query