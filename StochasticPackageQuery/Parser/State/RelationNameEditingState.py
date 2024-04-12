from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class RelationNameEditingState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_character_to_relation(char)
        return query