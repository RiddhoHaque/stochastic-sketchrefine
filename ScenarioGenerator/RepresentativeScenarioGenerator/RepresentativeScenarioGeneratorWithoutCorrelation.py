import numpy as np
from numpy.random import SFC64, SeedSequence, Generator

from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from SeedManager.SeedManager import SeedManager
from Utils.Relation_Prefixes import Relation_Prefixes


class RepresentativeScenarioGeneratorWithoutCorrelation(ScenarioGenerator):

    def __init__(
        self, 
        relation: str,
        attr: str,
        scenario_generator: ScenarioGenerator,
        base_predicate = '',
        duplicates = [],
        representatives = []
    ) -> None:
        self.__relation = relation
        self.__attribute = attr
        self.__base_predicate = base_predicate
        self.__duplicates = duplicates
        self.__representatives = representatives
        self.__scenario_generator = scenario_generator


    def __get_representatives(self):
        representative_relation = \
            Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX +\
            self.__relation
        
        sql = "SELECT representative_tuple_id " +\
            "FROM " + representative_relation + " WHERE attribute=" +\
            "'" + self.__attribute + "'"
        
        if len(self.__base_predicate) > 0:
            sql += ' AND ' + self.__base_predicate

        sql += ' ORDER BY partition_id;'

        PgConnection.Execute(sql)
        tuples = PgConnection.Fetch()

        representatives = []

        for tuple in tuples:
            representatives.append(tuple[0])
        
        return representatives    


    def generate_scenarios(
        self, seed: int, no_of_scenarios: int
    ) -> list[list[float]]:
        scenarios = []
        if len(self.__representatives) == 0:
            self.__representatives = \
                self.__get_representatives()

        scenarios = []
        duplicate_index = 0

        for representative in self.__representatives:
            scenario_generator = self.__scenario_generator(
                relation=self.__relation,
                base_predicate='id='+str(representative)
            )
            rep_scenarios = scenario_generator.generate_scenarios(
                seed=SeedManager.get_next_seed(),
                no_of_scenarios=no_of_scenarios*\
                    self.__duplicates[duplicate_index]
            )

            rep_scenarios = np.reshape(
                rep_scenarios, (self.__duplicates[duplicate_index],
                                no_of_scenarios))
            
            for scenario in rep_scenarios:
                scenarios.append(scenario)
            
            duplicate_index += 1
        
        return scenarios
    
