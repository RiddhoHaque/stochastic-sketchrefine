from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class ConstraintInequalitySettingState(State):

    def process(self, query: Query, char: chr) -> Query:
        query.set_constraint_inequality_sign(char)
        return query