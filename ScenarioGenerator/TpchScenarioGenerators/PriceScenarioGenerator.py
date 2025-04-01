import multiprocessing as mp
import numpy as np
from multiprocessing import Process, Queue
from numpy.random import SFC64, SeedSequence, Generator
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from Utils.Relation_Prefixes import Relation_Prefixes


class PriceScenarioGenerator(ScenarioGenerator):

    def __init__(self,
                 relation: str,
                 base_predicate = ''):
        self.__relation = relation
        self.__base_predicate = base_predicate
        self.__price_data = self.__get_price_attributes()
    
    def __get_price_attributes(self):
        if len(self.__base_predicate) == 0:
            self.__base_predicate = '1=1'
        sql_query = \
            'select price, price_mean, price_variance, '\
            'price_variance_coeff from ' + \
            self.__relation + ' where ' + self.__base_predicate + \
            ' order by id;'
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()

    def generate_scenarios(
        self, seed: int, no_of_scenarios: int,
    ) -> list[list[float]]:
        price_means = []
        price_variances = []
        rng = Generator(SFC64(SeedSequence(seed)))
        horizontal_bulk_generation = False
        if no_of_scenarios > len(self.__price_data):
            horizontal_bulk_generation = True
        prices = []
        for tuple in self.__price_data:
            price, price_mean,\
            price_variance,\
            price_variance_coeff = tuple
            price_means.append(
                price + price_mean
            )
            price_variances.append(
                price_variance*\
                price_variance_coeff
            )
            if horizontal_bulk_generation:
                prices.append(
                    rng.normal(
                        loc=price+price_mean,
                        scale=np.sqrt(
                            price_variance*\
                                price_variance_coeff),
                        size=no_of_scenarios
                    )
                )
        if horizontal_bulk_generation:
            return prices
        for _ in range(len(price_means)):
            prices.append([])
        for _ in range(no_of_scenarios):
            scenario = rng.normal(
                loc=price_means,
                scale=np.sqrt(price_variances)
            )
            for idx in range(len(scenario)):
                prices[idx].append(
                    scenario[idx]
                )
        return prices


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