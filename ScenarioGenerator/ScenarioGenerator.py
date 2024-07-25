import numpy as np
from PgConnection.PgConnection import PgConnection


class ScenarioGenerator:

    def __init__(self, 
                 relation: str,
                 base_predicate = '') -> None:
        self.__relation = relation
        self.__base_predicate = base_predicate
    
    def generate_scenarios(
        self, seed: int, no_of_scenarios: int
    ) -> list[list[float]]:
        scenarios = []
        for _ in range(1):
            scenarios.append([])
            for __ in range(no_of_scenarios):
                scenarios[0].append(0)
        return scenarios


