from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query

class ConstraintTailTypeEditingState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_character_to_constraint_tail_type(char)
        return query