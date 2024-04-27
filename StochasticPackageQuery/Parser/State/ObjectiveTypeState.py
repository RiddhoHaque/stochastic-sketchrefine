from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class ObjectiveTypeState(State):
    
    def process(self, query: Query, char: chr) -> Query:
        # 'a' from 'maximization
        if char == 'a':
            query.set_objective_type(is_maximization=True)
        else:
            query.set_objective_type(is_maximization=False)
        return query