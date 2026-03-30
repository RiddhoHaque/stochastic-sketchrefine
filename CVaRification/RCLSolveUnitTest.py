import unittest

from CVaRification.RCLSolve import RCLSolve
from StochasticPackageQuery.Constraints.CVaRConstraint.CVaRConstraint import CVaRConstraint
from StochasticPackageQuery.Constraints.ExpectedSumConstraint.ExpectedSumConstraint import ExpectedSumConstraint
from Utils.ObjectiveType import ObjectiveType


class _FakeObjective:

    def get_objective_type(self):
        return ObjectiveType.MAXIMIZATION


class _FakeQuery:

    def __init__(self, constraints):
        self.__constraints = constraints
        self.__objective = _FakeObjective()

    def get_constraints(self):
        return self.__constraints

    def get_objective(self):
        return self.__objective


class _FakeValidator:

    def __init__(
        self,
        expected_sum_feasible=None,
        cvar_feasible=None
    ):
        self.__expected_sum_feasible = expected_sum_feasible or {}
        self.__cvar_feasible = cvar_feasible or {}

    def get_expected_sum_constraint_feasibility(
        self, package_dict, expected_sum_constraint
    ):
        return self.__expected_sum_feasible.get(
            expected_sum_constraint, False
        )

    def get_cvar_constraint_feasibility(
        self, package_dict, cvar_constraint
    ):
        return self.__cvar_feasible.get(cvar_constraint, False)

    def get_var_constraint_feasibility(
        self, package_dict, var_constraint
    ):
        return False


class _DummyConstr:

    def __init__(self, rhs):
        self.rhs = rhs


class RCLSolveUnitTest(unittest.TestCase):

    def __make_solver(self, constraints, scenarios, validator):
        solver = object.__new__(RCLSolve)
        solver._RCLSolve__query = _FakeQuery(constraints)
        solver._RCLSolve__validator = validator
        solver._RCLSolve__ids = [1]
        solver._RCLSolve__scenarios = scenarios
        solver._RCLSolve__risk_to_lcvar_constraint_mapping = {}
        solver._RCLSolve__bisection_threshold = 0.5
        solver._RCLSolve__sampling_tolerance = 0.1
        solver._RCLSolve__approximation_bound = 0.1
        solver._RCLSolve__add_all_scenarios_if_possible = lambda no: None
        return solver

    def test_expected_sum_constraint_gets_search_bounds(self):
        esc = ExpectedSumConstraint()
        esc.set_attribute_name('gain')
        esc.set_inequality_sign('>')
        esc.set_sum_limit(10.0)

        solver = self.__make_solver(
            [esc],
            {'gain': [[14.0, 14.0]]},
            _FakeValidator(expected_sum_feasible={esc: False})
        )

        upper, lower, max_scenarios, min_scenarios, trivial = \
            solver._RCLSolve__get_bounds_for_risk_constraints(
                2, {1: 1.0}
            )

        self.assertEqual(lower, [10.0])
        self.assertEqual(upper, [28.0])
        self.assertEqual(min_scenarios, [1])
        self.assertEqual(max_scenarios, [1])
        self.assertEqual(trivial, [])

    def test_expected_sum_and_cvar_share_constraint_index_space(self):
        esc = ExpectedSumConstraint()
        esc.set_attribute_name('gain')
        esc.set_inequality_sign('>')
        esc.set_sum_limit(10.0)

        cvar = CVaRConstraint()
        cvar.set_attribute_name('risk_gain')
        cvar.set_inequality_sign('>')
        cvar.set_sum_limit(8.0)
        cvar.set_percentage_of_scenarios(50.0)
        cvar.set_tail_type('l')

        solver = self.__make_solver(
            [esc, cvar],
            {
                'gain': [[10.0, 10.0]],
                'risk_gain': [[9.0, 9.0, 9.0, 9.0]],
            },
            _FakeValidator(
                expected_sum_feasible={esc: True},
                cvar_feasible={cvar: False}
            )
        )
        solver._RCLSolve__get_linearized_cvar_among_optimization_scenarios = \
            lambda package, constraint, no_of_scenarios: 12.0

        upper, lower, max_scenarios, min_scenarios, trivial = \
            solver._RCLSolve__get_bounds_for_risk_constraints(
                4, {1: 1.0}
            )

        self.assertEqual(lower, [10.0, 9.0])
        self.assertEqual(upper, [20.0, 24.0])
        self.assertEqual(min_scenarios, [1, 2])
        self.assertEqual(max_scenarios, [1, 4])
        self.assertEqual(trivial, [0])

    def test_threshold_search_keeps_feasible_lower_bound_when_tighter_midpoint_has_no_package(self):
        esc = ExpectedSumConstraint()
        esc.set_attribute_name('gain')
        esc.set_inequality_sign('>')
        esc.set_sum_limit(10.0)

        solver = self.__make_solver(
            [esc],
            {'gain': [[12.0, 12.0]]},
            _FakeValidator(expected_sum_feasible={esc: True})
        )
        seen_rhs = []

        def fake_model_setup(
            no_of_scenarios,
            no_of_scenarios_to_consider,
            probabilistically_constrained=False,
            cvar_lower_bounds=None,
            trivial_constraints=None
        ):
            rhs = cvar_lower_bounds[0]
            seen_rhs.append(rhs)
            solver._RCLSolve__risk_to_lcvar_constraint_mapping = {
                esc: _DummyConstr(rhs)
            }

        def fake_get_package():
            rhs = solver._RCLSolve__risk_to_lcvar_constraint_mapping[esc].rhs
            seen_rhs.append(rhs)
            if rhs <= 12.0:
                return {1: 1.0}
            return None

        solver._RCLSolve__model_setup = fake_model_setup
        solver._RCLSolve__get_package = fake_get_package
        solver._RCLSolve__get_package_with_indices = lambda: {1: 1.0}
        solver._RCLSolve__is_objective_value_relative_diff_high = \
            lambda package, package_with_indices, no_of_scenarios: (False, 5.0)
        solver._RCLSolve__is_objective_value_enough = \
            lambda validation_objective_value, objective_upper_bound: False

        result = solver._RCLSolve__cvar_threshold_search(
            no_of_scenarios=2,
            cvar_upper_bounds=[14.0],
            cvar_lower_bounds=[10.0],
            trivial_constraints=[],
            objective_upper_bound=100.0,
            no_of_scenarios_to_consider=[1],
            is_model_setup=False,
            can_add_scenarios=False
        )

        self.assertEqual(result.get_cvar_thresholds()[0], 12.0)
        self.assertIn(12.0, seen_rhs)
        self.assertIn(13.0, seen_rhs)

    def test_lower_anchor_repair_widens_expected_sum_constraint(self):
        esc = ExpectedSumConstraint()
        esc.set_attribute_name('gain')
        esc.set_inequality_sign('>')
        esc.set_sum_limit(10.0)

        solver = self.__make_solver(
            [esc],
            {'gain': [[12.0, 12.0]]},
            _FakeValidator()
        )

        def fake_model_setup(
            no_of_scenarios,
            no_of_scenarios_to_consider,
            probabilistically_constrained=False,
            cvar_lower_bounds=None,
            trivial_constraints=None
        ):
            solver._RCLSolve__risk_to_lcvar_constraint_mapping = {
                esc: _DummyConstr(cvar_lower_bounds[0])
            }

        solver._RCLSolve__model_setup = fake_model_setup
        solver._RCLSolve__get_package = lambda: {1: 1.0}
        solver._RCLSolve__get_package_with_indices = lambda: {1: 1.0}
        solver._RCLSolve__is_objective_value_relative_diff_high = \
            lambda package, package_with_indices, no_of_scenarios: (False, 0.0)
        solver._RCLSolve__validator.get_expected_sum_constraint_feasibility = \
            lambda package_dict, constraint: \
                solver._RCLSolve__risk_to_lcvar_constraint_mapping[
                    esc].rhs <= 8.0

        lower, is_model_setup, needs_more_scenarios = \
            solver._RCLSolve__repair_lower_bound_anchors(
                no_of_scenarios=2,
                cvar_upper_bounds=[14.0],
                cvar_lower_bounds=[10.0],
                trivial_constraints=[],
                no_of_scenarios_to_consider=[1],
                can_add_scenarios=False
            )

        self.assertEqual(lower, [6.0])
        self.assertTrue(is_model_setup)
        self.assertFalse(needs_more_scenarios)

    def test_lower_anchor_repair_keeps_current_bounds_when_package_none(self):
        esc = ExpectedSumConstraint()
        esc.set_attribute_name('gain')
        esc.set_inequality_sign('>')
        esc.set_sum_limit(10.0)

        solver = self.__make_solver(
            [esc],
            {'gain': [[12.0, 12.0]]},
            _FakeValidator()
        )

        def fake_model_setup(
            no_of_scenarios,
            no_of_scenarios_to_consider,
            probabilistically_constrained=False,
            cvar_lower_bounds=None,
            trivial_constraints=None
        ):
            solver._RCLSolve__risk_to_lcvar_constraint_mapping = {
                esc: _DummyConstr(cvar_lower_bounds[0])
            }

        def fake_get_package():
            rhs = solver._RCLSolve__risk_to_lcvar_constraint_mapping[esc].rhs
            if rhs >= 10.0:
                return {1: 1.0}
            return None

        solver._RCLSolve__model_setup = fake_model_setup
        solver._RCLSolve__get_package = fake_get_package
        solver._RCLSolve__get_package_with_indices = lambda: {1: 1.0}
        solver._RCLSolve__is_objective_value_relative_diff_high = \
            lambda package, package_with_indices, no_of_scenarios: (False, 0.0)
        solver._RCLSolve__validator.get_expected_sum_constraint_feasibility = \
            lambda package_dict, constraint: False

        lower, is_model_setup, needs_more_scenarios = \
            solver._RCLSolve__repair_lower_bound_anchors(
                no_of_scenarios=2,
                cvar_upper_bounds=[14.0],
                cvar_lower_bounds=[10.0],
                trivial_constraints=[],
                no_of_scenarios_to_consider=[1],
                can_add_scenarios=False
            )

        self.assertEqual(lower, [6.0])
        self.assertTrue(is_model_setup)
        self.assertFalse(needs_more_scenarios)

    def test_lower_anchor_repair_does_not_widen_trivial_constraints(self):
        trivial_esc = ExpectedSumConstraint()
        trivial_esc.set_attribute_name('gain')
        trivial_esc.set_inequality_sign('>')
        trivial_esc.set_sum_limit(10.0)

        nontrivial_esc = ExpectedSumConstraint()
        nontrivial_esc.set_attribute_name('gain')
        nontrivial_esc.set_inequality_sign('>')
        nontrivial_esc.set_sum_limit(10.0)

        solver = self.__make_solver(
            [trivial_esc, nontrivial_esc],
            {'gain': [[12.0, 12.0]]},
            _FakeValidator()
        )

        def fake_model_setup(
            no_of_scenarios,
            no_of_scenarios_to_consider,
            probabilistically_constrained=False,
            cvar_lower_bounds=None,
            trivial_constraints=None
        ):
            solver._RCLSolve__risk_to_lcvar_constraint_mapping = {
                trivial_esc: _DummyConstr(cvar_lower_bounds[0]),
                nontrivial_esc: _DummyConstr(cvar_lower_bounds[1])
            }

        solver._RCLSolve__model_setup = fake_model_setup
        solver._RCLSolve__get_package = lambda: {1: 1.0}
        solver._RCLSolve__get_package_with_indices = lambda: {1: 1.0}
        solver._RCLSolve__is_objective_value_relative_diff_high = \
            lambda package, package_with_indices, no_of_scenarios: (False, 0.0)

        def expected_sum_feasible(package_dict, constraint):
            rhs = solver._RCLSolve__risk_to_lcvar_constraint_mapping[
                constraint].rhs
            if constraint is trivial_esc:
                return False
            return rhs <= 8.0

        solver._RCLSolve__validator.get_expected_sum_constraint_feasibility = \
            expected_sum_feasible

        lower, is_model_setup, needs_more_scenarios = \
            solver._RCLSolve__repair_lower_bound_anchors(
                no_of_scenarios=2,
                cvar_upper_bounds=[20.0, 14.0],
                cvar_lower_bounds=[10.0, 10.0],
                trivial_constraints=[0],
                no_of_scenarios_to_consider=[1, 1],
                can_add_scenarios=False
            )

        self.assertEqual(lower, [10.0, 6.0])
        self.assertTrue(is_model_setup)
        self.assertFalse(needs_more_scenarios)


if __name__ == '__main__':
    unittest.main()
