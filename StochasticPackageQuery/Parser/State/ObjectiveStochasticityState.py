from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Query import Query


class ObjectiveStochasticityState(State):

    def process(self, query: Query, char: chr) -> Query:
        #'e' from expected sum
        if char == 'e':
            query.set_objective_stochasticity(is_stochastic=True)
        else:
            query.set_objective_stochasticity(is_stochastic=False)
        return query