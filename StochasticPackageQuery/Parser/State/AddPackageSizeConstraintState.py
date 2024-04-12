from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class AddPackageSizeConstraint(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_package_size_constraint()
        return query