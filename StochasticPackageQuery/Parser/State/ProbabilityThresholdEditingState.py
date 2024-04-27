from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class ProbabilityThresholdEditingState(State):
    def process(self, query: Query, char: chr) -> Query:
        query.add_character_to_constraint_probability_threshold(char)
        return query