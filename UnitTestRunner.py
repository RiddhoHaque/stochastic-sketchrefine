from StochasticPackageQuery.Constraints.ConstraintUnitTest import ConstraintUnitTest
from StochasticPackageQuery.Constraints.RepeatConstraint.RepeatConstraintUnitTest import RepeatConstraintUnitTest
from StochasticPackageQuery.Constraints.PackageSizeConstraint.PackageSizeConstraintUnitTest import PackageSizeConstraintUnitTest
from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraintUnitTest import DeterministicConstraintUnitTest
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraintUnitTest import ExpectedSumConstraintUnitTest
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraintUnitTest import VaRConstraintUnitTest
from StochasticPackageQuery.Objective.ObjectiveUnitTest import ObjectiveUnitTest
from StochasticPackageQuery.QueryUnitTest import QueryUnitTest
from StochasticPackageQuery.Parser.ParserUnitTest import ParserUnitTest
from StochasticPackageQuery.Parser.State.StateUnitTest import StateUnitTest
from StochasticPackageQuery.Parser.Transition.TransitionUnitTest import TransitionUnitTest
from Utils.UtilsUnitTest import UtilsUnitTest


def UnitTestRunner():
    RepeatConstraintUnitTest().main()
    PackageSizeConstraintUnitTest().main()
    DeterministicConstraintUnitTest().main()
    ExpectedSumConstraintUnitTest().main()
    VaRConstraintUnitTest().main()
    ObjectiveUnitTest().main()
    ConstraintUnitTest().main()
    UtilsUnitTest().main()
    QueryUnitTest().main()
    TransitionUnitTest().main()
    StateUnitTest().main()
    ParserUnitTest().main()
    print('All unit tests passed')
