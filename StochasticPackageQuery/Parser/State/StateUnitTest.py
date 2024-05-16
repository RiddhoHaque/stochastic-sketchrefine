import unittest
from StochasticPackageQuery.Parser.State.AddDeterministicConstraintState import AddDeterministicConstraintState
from StochasticPackageQuery.Parser.State.AddExpectedSumConstraintState import AddExpectedSumConstraintState
from StochasticPackageQuery.Parser.State.AddPackageSizeConstraintState import AddPackageSizeConstraintState
from StochasticPackageQuery.Parser.State.AddRepeatConstraintState import AddRepeatConstraintState
from StochasticPackageQuery.Parser.State.BasePredicateEditingState import BasePredicateEditingState
from StochasticPackageQuery.Parser.State.ConstraintAttributeNameEditingState import ConstraintAttributeNameEditingState
from StochasticPackageQuery.Parser.State.ConstraintInequalitySettingState import ConstraintInequalitySettingState
from StochasticPackageQuery.Parser.State.ConstraintPercentageEditingState import ConstraintPercentageEditingState
from StochasticPackageQuery.Parser.State.ConstraintSumLimitSettingState import ConstraintSumLimitSettingState
from StochasticPackageQuery.Parser.State.ConstraintTailTypeEditingState import ConstraintTailTypeEditingState
from StochasticPackageQuery.Parser.State.ObjectiveAttributeNameEditingState import ObjectiveAttributeNameEditingState
from StochasticPackageQuery.Parser.State.ObjectiveStochasticityState import ObjectiveStochasticityState
from StochasticPackageQuery.Parser.State.ObjectiveTypeState import ObjectiveTypeState
from StochasticPackageQuery.Parser.State.PackageAliasEditingState import PackageAliasEditingState
from StochasticPackageQuery.Parser.State.PackageSizeLimitEditingState import PackageSizeLimitEditingState
from StochasticPackageQuery.Parser.State.ProbabilityThresholdEditingState import ProbabilityThresholdEditingState
from StochasticPackageQuery.Parser.State.ProjectedAttributeEditingState import ProjectedAttributeEditingState
from StochasticPackageQuery.Parser.State.RelationAliasEditingState import RelationAliasEditingState
from StochasticPackageQuery.Parser.State.RelationNameEditingState import RelationNameEditingState
from StochasticPackageQuery.Parser.State.RepetitionLimitEditingState import RepetitionLimitEditingState
from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Parser.State.TurnToVaRConstraintState import TurnToVaRConstraintState
from StochasticPackageQuery.Parser.State.TurnToCVaRConstraintState import TurnToCVaRConstraintState
from StochasticPackageQuery.Parser.Transition.Transition import Transition
from StochasticPackageQuery.Query import Query
from Utils.RelationalOperators import RelationalOperators
from Utils.ObjectiveType import ObjectiveType
from Utils.Stochasticity import Stochasticity
from Utils.TailType import TailType


class StateUnitTest(unittest.TestCase):

    def test_process_keeps_query_unchanged(self):
        state = State()
        query = Query()
        self.assertEqual(state.process(query, 'a'), query)
        
    def test_transitions_added_correctly(self):
        state = State()
        next_state = State()
        transition1 = Transition('a', next_state)
        transition2 = Transition('b', next_state)
        state.add_transition(transition1)
        state.add_transition(transition2)
        with self.assertRaises(Exception):
            state.add_transition(Transition('a', next_state))
        self.assertEqual(len(state.get_transitions()), 2)
        transition3 = Transition('a', next_state,
                                 anything_but_transition=True)
        state.add_transition(transition3)
        self.assertEqual(state.get_transitions()[0], transition1)
        self.assertEqual(state.get_transitions()[1], transition2)
        self.assertEqual(state.get_transitions()[2], transition3)

    def test_get_next_state_correctly(self):
        state = State()
        next_state1 = State()
        next_state2 = State()
        transition1 = Transition('a', next_state1)
        transition2 = Transition('b', next_state2)
        state.add_transition(transition1)
        state.add_transition(transition2)
        self.assertEqual(state.get_next_state('a'), next_state1)
        self.assertEqual(state.get_next_state('b'), next_state2)
        with self.assertRaises(Exception):
            state.get_next_state('c')

    def test_add_deterministic_constraint_state(self):
        state = AddDeterministicConstraintState()
        query = Query()
        query = state.process(query, 'a')
        self.assertEqual(len(query.get_constraints()), 1)
        self.assertTrue(query.get_constraints()[-1].is_deterministic_constraint())
    
    def test_add_expected_sum_constraint_state(self):
        state = AddExpectedSumConstraintState()
        query = Query()
        query = state.process(query, 'a')
        self.assertEqual(len(query.get_constraints()), 1)
        self.assertTrue(query.get_constraints()[-1].is_expected_sum_constraint())

    def test_add_package_size_constraint_state(self):
        state = AddPackageSizeConstraintState()
        query = Query()
        query = state.process(query, 'a')
        self.assertEqual(len(query.get_constraints()), 1)
        self.assertTrue(query.get_constraints()[0].is_package_size_constraint())

    def test_base_predicate_editing_state(self):
        state = BasePredicateEditingState()
        query = Query()
        base_predicate = "where gluten='free'"
        for character in base_predicate:
            query = state.process(query, character)
        self.assertEqual(query.get_base_predicate(), base_predicate)
    
    def test_constraint_attribute_name_editing_state(self):
        state = ConstraintAttributeNameEditingState()
        query = Query()
        query.add_deterministic_constraint()
        attribute_name = 'attr'
        for character in attribute_name:
            query = state.process(query, character)
        self.assertEqual(query.get_constraints()[-1
                            ].get_attribute_name(),
                         attribute_name)
        
    def test_constraint_inequality_setting_state(self):
        state = ConstraintInequalitySettingState()
        query = Query()
        query.add_deterministic_constraint()
        query = state.process(query, '>')
        self.assertEqual(query.get_constraints()[-1
                            ].get_inequality_sign(),
                        RelationalOperators.GREATER_THAN_OR_EQUAL_TO)

    def test_constraint_sum_limit_setting_state(self):
        state = ConstraintSumLimitSettingState()
        query = Query()
        query.add_deterministic_constraint()
        query = state.process(query, '1')
        self.assertEqual(query.get_constraints()[-1
                            ].get_sum_limit(), 1)
    
    def test_objective_attribute_name_editting_state(self):
        state = ObjectiveAttributeNameEditingState()
        query = Query()
        query = state.process(query, 'a')
        query = state.process(query, 't')
        query = state.process(query, 't')
        query = state.process(query, 'r')
        self.assertEqual(query.get_objective().get_attribute_name(), 'attr')

    def test_objective_type_state(self):
        state = ObjectiveTypeState()
        query = Query()
        query = state.process(query, 'a')
        self.assertEqual(query.get_objective().get_objective_type(),
                         ObjectiveType.MAXIMIZATION)
    
    def test_objective_stochasticity_state(self):
        state = ObjectiveStochasticityState()
        query = Query()
        query = state.process(query, 'e')
        self.assertEqual(query.get_objective().get_stochasticity(),
                         Stochasticity.STOCHASTIC)        

    def test_package_alias_editing_state(self):
        state = PackageAliasEditingState()
        query = Query()
        query = state.process(query, 'p')
        self.assertEqual(query.get_package_alias(), 'p')

    def test_package_size_limit_editing_state(self):
        state = PackageSizeLimitEditingState()
        query = Query()
        query.add_package_size_constraint()
        query = state.process(query, '5')
        self.assertEqual(query.get_constraints()[
            -1].get_package_size_limit(), 5)

    def test_probability_threshold_editing_state(self):
        state = ProbabilityThresholdEditingState()
        query = Query()
        query.add_var_constraint()
        query = state.process(query, '0')
        query = state.process(query, '.')
        query = state.process(query, '9')
        query = state.process(query, '5')
        self.assertEqual(query.get_constraints()[
            -1].get_probability_threshold(), 0.95)

    def test_projected_attribute_editing_state(self):
        state = ProjectedAttributeEditingState()
        query = Query()
        query = state.process(query, '*')
        self.assertEqual(query.get_projected_attributes(),
                         '*')

    def test_relation_alias_editing_state(self):
        state = RelationAliasEditingState()
        query = Query()
        query = state.process(query, 't')
        self.assertEqual(query.get_relation_alias(),
                         't')

    def test_relation_name_editing_state(self):
        state = RelationNameEditingState()
        query = Query()
        query = state.process(query, 't')
        self.assertEqual(query.get_relation(),
                         't') 

    def test_repetition_limit_editing_state(self):
        state = RepetitionLimitEditingState()
        query = Query()
        query.add_repeat_constraint()
        query = state.process(query, '2')
        self.assertEqual(query.get_constraints()[
                        -1].get_repetition_limit(), 2)
    
    def test_constraint_tail_type_editing_state(self):
        state = ConstraintTailTypeEditingState()
        query = Query()
        query.add_cvar_constraint()
        query = state.process(query, 'l')
        self.assertEqual(query.get_constraints()[
            -1].get_tail_type(), TailType.LOWEST)

    def test_constraint_percentage_editing_state(self):
        state = ConstraintPercentageEditingState()
        query = Query()
        query.add_cvar_constraint()
        query = state.process(query, '6')
        self.assertEqual(query.get_constraints()[
            -1].get_percentage_of_scenarios(), 6)

    def test_turn_to_var_constraint_state(self):
        state = TurnToVaRConstraintState()
        query = Query()
        query.add_deterministic_constraint()
        query.add_character_to_attribute_name('a')
        query.set_constraint_inequality_sign('<')
        query.add_character_to_constraint_sum_limit('1')
        query = state.process(query, 'a')
        self.assertEqual(len(query.get_constraints()), 1)
        self.assertTrue(query.get_constraints()[0].is_var_constraint())
    
    def test_turn_to_cvar_constraint_state(self):
        state = TurnToCVaRConstraintState()
        query = Query()
        query.add_expected_sum_constraint()
        query.add_character_to_attribute_name('a')
        query.set_constraint_inequality_sign('<')
        query.add_character_to_constraint_sum_limit('1')
        query = state.process(query, 'a')
        self.assertEqual(len(query.get_constraints()), 1)
        self.assertTrue(query.get_constraints()[0].is_cvar_constraint())


    def main(self):
        self.test_process_keeps_query_unchanged()
        self.test_transitions_added_correctly()
        self.test_get_next_state_correctly()
        self.test_add_deterministic_constraint_state()
        self.test_add_expected_sum_constraint_state()
        self.test_add_package_size_constraint_state()
        self.test_base_predicate_editing_state()
        self.test_constraint_attribute_name_editing_state()
        self.test_constraint_inequality_setting_state()
        self.test_constraint_sum_limit_setting_state()
        self.test_objective_attribute_name_editting_state()
        self.test_objective_type_state()
        self.test_objective_stochasticity_state()
        self.test_package_alias_editing_state()
        self.test_package_size_limit_editing_state()
        self.test_probability_threshold_editing_state()
        self.test_projected_attribute_editing_state()
        self.test_relation_alias_editing_state()
        self.test_relation_name_editing_state()
        self.test_repetition_limit_editing_state()
        self.test_constraint_percentage_editing_state()
        self.test_constraint_tail_type_editing_state()
        self.test_turn_to_var_constraint_state()
        self.test_turn_to_cvar_constraint_state()

