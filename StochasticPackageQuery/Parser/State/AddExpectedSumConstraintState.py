from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class AddExpectedSumConstraintState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_expected_sum_constraint()
        return query