from PgConnection.PgConnectionUnitTest import PgConnectionUnitTest
from OfflinePreprocessing.MeanAbsoluteDistanceUnitTest import MeanAbsoluteDistanceUnitTest
from OfflinePreprocessing.PivotScanUnitTest import PivotScanUnitTest
from ScenarioGenerator.ScenarioGeneratorUnitTest import ScenarioGeneratorUnitTest
from ScenarioGenerator.TpchScenarioGenerators.PriceScenarioGeneratorUnitTest import PriceScenarioGeneratorUnitTest
from ScenarioGenerator.TpchScenarioGenerators.QuantityScenarioGeneratorUnitTest import QuantityScenarioGeneratorUnitTest
from ScenarioGenerator.PorfolioScenarioGenerator.GainScenarioGeneratorUnitTest import GainScenarioGeneratorUnitTest
from StochasticPackageQuery.Constraints.ConstraintUnitTest import ConstraintUnitTest
from StochasticPackageQuery.Constraints.RepeatConstraint.RepeatConstraintUnitTest import RepeatConstraintUnitTest
from StochasticPackageQuery.Constraints.PackageSizeConstraint.PackageSizeConstraintUnitTest import PackageSizeConstraintUnitTest
from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraintUnitTest import DeterministicConstraintUnitTest
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraintUnitTest import ExpectedSumConstraintUnitTest
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraintUnitTest import VaRConstraintUnitTest
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraintUnitTest import CVaRConstraintUnitTest
from StochasticPackageQuery.Objective.ObjectiveUnitTest import ObjectiveUnitTest
from StochasticPackageQuery.QueryUnitTest import QueryUnitTest
from StochasticPackageQuery.Parser.ParserUnitTest import ParserUnitTest
from StochasticPackageQuery.Parser.State.StateUnitTest import StateUnitTest
from StochasticPackageQuery.Parser.Transition.TransitionUnitTest import TransitionUnitTest
from Utils.HeapUnitTest import HeapUnitTest
from Utils.UtilsUnitTest import UtilsUnitTest
from ValueGenerator.ValueGeneratorUnitTest import ValueGeneratorUnitTest


def UnitTestRunner():
    RepeatConstraintUnitTest().main()
    PackageSizeConstraintUnitTest().main()
    DeterministicConstraintUnitTest().main()
    ExpectedSumConstraintUnitTest().main()
    VaRConstraintUnitTest().main()
    CVaRConstraintUnitTest().main()
    ObjectiveUnitTest().main()
    ConstraintUnitTest().main()
    UtilsUnitTest().main()
    QueryUnitTest().main()
    TransitionUnitTest().main()
    StateUnitTest().main()
    ParserUnitTest().main()
    PgConnectionUnitTest().main()
    ScenarioGeneratorUnitTest().main()
    PriceScenarioGeneratorUnitTest().main()
    QuantityScenarioGeneratorUnitTest().main()
    GainScenarioGeneratorUnitTest().main()
    ValueGeneratorUnitTest().main()
    MeanAbsoluteDistanceUnitTest().main()
    PivotScanUnitTest().main()
    HeapUnitTest().main()
    print('All unit tests passed')
