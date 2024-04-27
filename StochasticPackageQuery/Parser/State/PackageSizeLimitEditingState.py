from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class PackageSizeLimitEditingState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_digit_to_package_size_constraint(int(char))
        return query