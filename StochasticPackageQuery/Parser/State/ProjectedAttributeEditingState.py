from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class ProjectedAttributeEditingState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_character_to_projected_attributes(char)
        return query