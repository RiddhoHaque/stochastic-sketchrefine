from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class CommitCVaRConstraintProbabilityState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.commit_cvar_constraint_probability()
        return query
