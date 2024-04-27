from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class TurnToVaRConstraintState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.convert_final_deterministic_constraint_to_var_constraint()
        return query