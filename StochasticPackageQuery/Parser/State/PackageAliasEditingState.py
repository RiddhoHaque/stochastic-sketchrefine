from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class PackageAliasEditingState(State):

    def process(self, query: Query, char: chr) -> Query:
        return query.add_character_to_package_alias(char)