from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class TurnToCVaRConstraintState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.convert_final_expected_sum_constraint_to_cvar_constraint()
        return query