import math
import numpy as np
import time


from DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGenerator import RepresentativeScenarioGenerator
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGeneratorWithoutCorrelation import RepresentativeScenarioGeneratorWithoutCorrelation
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraint import VaRConstraint
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from StochasticPackageQuery.Query import Query
from Utils.RelationalOperators import RelationalOperators
from Utils.Relation_Prefixes import Relation_Prefixes
from Utils.TailType import TailType
from Utils.ObjectiveType import ObjectiveType


class RefineValidator:

    def __init__(
        self, query: Query,
        dbInfo: DbInfo,
        no_of_validation_scenarios: int,
        partition_group,
        chosen_tuples_with_multiplicities,
        remaining_groups_with_multiplicities,
        sizes
    ):
        self.__query = query
        self.__dbInfo = dbInfo
        self.__no_of_validation_scenarios =\
            no_of_validation_scenarios
        self.__partition_group =\
            partition_group
        
        self.__chosen_tuples = []
        for tuple, _ in chosen_tuples_with_multiplicities:
            self.__chosen_tuples.append(tuple)
        
        self.__remaining_groups = []
        for group, _ in remaining_groups_with_multiplicities:
            self.__remaining_groups.append(group)
        
        self.__sizes = sizes