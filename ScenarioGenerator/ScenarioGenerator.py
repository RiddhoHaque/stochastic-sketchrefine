import numpy as np
from PgConnection.PgConnection import PgConnection
from Utils.Relation_Prefixes import Relation_Prefixes


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


    def generate_scenarios_from_partition(
        self, seed: int, no_of_scenarios: int,
        partition_id: int
    ) -> list[list[float]]:
        self.__relation == self.__relation +\
            ' AS r INNER JOIN ' + \
                Relation_Prefixes.PARTITION_RELATION_PREFIX +\
                    self.__relation + ' AS p ON r.id=p.tuple_id'

        if len(self.__base_predicate) > 0:
            self.__base_predicate += ' AND '
        self.__base_predicate += 'p.partition_id = ' + str(
            partition_id
        )

        return self.generate_scenarios(
            seed, no_of_scenarios
        ) 
