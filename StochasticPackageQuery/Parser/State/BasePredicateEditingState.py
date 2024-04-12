from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class BasePredicateEditingState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_character_to_base_predicate(char)
        return query