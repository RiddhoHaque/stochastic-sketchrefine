import numpy as np
from numpy.random import SFC64, SeedSequence, Generator
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from Utils.Relation_Prefixes import Relation_Prefixes


class QuantityScenarioGenerator(ScenarioGenerator):

    def __init__(self,
                 relation: str,
                 base_predicate = ''):
        self.__relation = relation
        self.__base_predicate = base_predicate
        self.__quantity_data = self.__get_quantity_attributes()

    def __get_quantity_attributes(self):
        if len(self.__base_predicate) == 0:
            self.__base_predicate = '1=1'
        sql_query = \
            'select quantity, quantity_mean, quantity_variance, '\
            'quantity_variance_coeff from ' + \
            self.__relation + ' where ' + self.__base_predicate + \
            ' order by id;'
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()
    
    def generate_scenarios(self, seed, no_of_scenarios):
        quantity_means = []
        quantity_variances = []
        rng = Generator(SFC64(SeedSequence(seed)))
        horizontal_bulk_generation = False
        if no_of_scenarios > len(self.__quantity_data):
            horizontal_bulk_generation = True
        quantities = []
        for tuple in self.__quantity_data:
            quantity, quantity_mean,\
            quantity_variance,\
            quantity_variance_coeff = tuple
            quantity_means.append(
                quantity + quantity_mean
            )
            quantity_variances.append(
                quantity_variance*\
                    quantity_variance_coeff
            )
            if horizontal_bulk_generation:
                quantities.append(
                    rng.normal(
                        loc=quantity+quantity_mean,
                        scale=np.sqrt(
                            quantity_variance*\
                                quantity_variance_coeff),
                        size=no_of_scenarios
                    )
                )
        if horizontal_bulk_generation:
            return quantities
        for _ in range(len(quantity_means)):
            quantities.append([])
        for _ in range(no_of_scenarios):
            scenario = rng.normal(
                loc=quantity_means,
                scale=np.sqrt(quantity_variances)
            )
            for idx in range(len(scenario)):
                quantities[idx].append(
                    scenario[idx]
                )
        return quantities


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