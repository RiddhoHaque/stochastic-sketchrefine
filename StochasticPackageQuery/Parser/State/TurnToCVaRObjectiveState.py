from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class TurnToCVaRObjectiveState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.set_objective_as_cvar()
        return query
