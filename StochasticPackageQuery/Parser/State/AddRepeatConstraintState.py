from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class AddRepeatConstraintState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_repeat_constraint()
        return query
