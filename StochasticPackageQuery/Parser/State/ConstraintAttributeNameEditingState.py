from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class ConstraintAttributeNameEditingState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_character_to_attribute_name(char)
        return query