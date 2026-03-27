from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class ObjectivePercentageEditingState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.add_character_to_objective_percentage_of_scenarios(char)
        return query
