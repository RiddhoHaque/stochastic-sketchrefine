from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class CVaRProbabilityEditingState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.accumulate_cvar_probability(char)
        return query
