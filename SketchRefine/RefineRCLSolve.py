import gurobipy as gp
from gurobipy import GRB
import numpy as np
import psutil

from CVaRification.CVaRificationSearchResults import CVaRificationSearchResults
from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OptimizationMetrics.OptimizationMetrics import OptimizationMetrics
from PgConnection.PgConnection import PgConnection
from SeedManager.SeedManager import SeedManager
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraint import VaRConstraint
from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraint import DeterministicConstraint
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from StochasticPackageQuery.Constraints.PackageSizeConstraint.PackageSizeConstraint import PackageSizeConstraint
from StochasticPackageQuery.Objective.Objective import Objective
from StochasticPackageQuery.Query import Query
from Utils.GurobiLicense import GurobiLicense
from Utils.Heap import Heap
from Utils.ObjectiveType import ObjectiveType
from Utils.RelationalOperators import RelationalOperators
from Utils.Stochasticity import Stochasticity
from Utils.TailType import TailType
from Validator.Validator import Validator
from ValueGenerator.ValueGenerator import ValueGenerator


class RefineRCLSolve:

    def __init__(self, query: Query,
                 linear_relaxation: bool,
                 scenarios: dict[str, list[list[float]]],
                 values: dict[str, list[float]],
                 init_no_of_scenarios: int,
                 no_of_validation_scenarios: int,
                 approximation_bound: float,
                 sampling_tolerance: float,
                 bisection_threshold: float):
        self.__query = query
        self.__gurobi_