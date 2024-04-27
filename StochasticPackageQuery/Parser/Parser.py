from StochasticPackageQuery.Parser.State.AddDeterministicConstraintState import AddDeterministicConstraintState
from StochasticPackageQuery.Parser.State.AddExpectedSumConstraintState import AddExpectedSumConstraintState
from StochasticPackageQuery.Parser.State.AddPackageSizeConstraintState import AddPackageSizeConstraintState
from StochasticPackageQuery.Parser.State.AddRepeatConstraintState import AddRepeatConstraintState
from StochasticPackageQuery.Parser.State.BasePredicateEditingState import BasePredicateEditingState
from StochasticPackageQuery.Parser.State.ConstraintAttributeNameEditingState import ConstraintAttributeNameEditingState
from StochasticPackageQuery.Parser.State.ConstraintInequalitySettingState import ConstraintInequalitySettingState
from StochasticPackageQuery.Parser.State.ConstraintSumLimitSettingState import ConstraintSumLimitSettingState
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
from StochasticPackageQuery.Parser.State.RubberDuckState import RubberDuckState
from StochasticPackageQuery.Parser.State.State import State
from StochasticPackageQuery.Parser.State.TurnToVaRConstraintState import TurnToVaRConstraintState
from StochasticPackageQuery.Parser.Transition.Transition import Transition
from StochasticPackageQuery.Query import Query


class Parser:

    __digits = ['0', '1', '2', '3', '4', '5', '6', '7',
              '8', '9']
    
    __numericals = ['-', '.', '0', '1', '2', '3', '4',
                  '5', '6', '7', '8', '9']

    def __expect_phrase(self, starting_state: State,
                    word: str) -> State:
        current_state = starting_state
        for character in word:
            new_state = State()
            current_state.add_transition(Transition(
                character, new_state))
            current_state = new_state
        return current_state


    def __init__(self) -> None:
        self.__init_state = State()
        self.__init_state.add_transition(
            Transition(' ', self.__init_state)
        )
        intro_parsed_state = self.__expect_phrase(
            self.__init_state, 'select package(')
        get_projected_attributes_state = \
            ProjectedAttributeEditingState()
        intro_parsed_state.add_transition(
            Transition(')', get_projected_attributes_state,
                       anything_but_transition=True))
        get_projected_attributes_state.add_transition(
            Transition(')', get_projected_attributes_state,
                       anything_but_transition=True))
        projected_attributes_taken_state = State()
        get_projected_attributes_state.add_transition(
            Transition(')', projected_attributes_taken_state)
        )
        space_after_projected_attributes = State()
        projected_attributes_taken_state.add_transition(
            Transition(' ', space_after_projected_attributes)
        )
        package_alias_exists_detected_state = State()
        space_after_projected_attributes.add_transition(
            Transition('a', package_alias_exists_detected_state)
        )
        as_keyword_read_state = State()
        package_alias_exists_detected_state.add_transition(
            Transition('s', as_keyword_read_state)
        )
        ready_for_package_alias_state = State()
        as_keyword_read_state.add_transition(
            Transition(' ', ready_for_package_alias_state)
        )
        ready_for_package_alias_state.add_transition(
            Transition(' ', ready_for_package_alias_state)
        )
        package_alias_reading_state = PackageAliasEditingState()
        ready_for_package_alias_state.add_transition(
            Transition(' ', package_alias_reading_state,
                       anything_but_transition=True)
        )
        package_alias_reading_state.add_transition(
            Transition(' ', package_alias_reading_state,
                       anything_but_transition=True)
        )
        package_alias_reading_state.add_transition(
            Transition(' ', space_after_projected_attributes)
        )
        space_after_projected_attributes.add_transition(
            Transition(' ', space_after_projected_attributes)
        )
        from_keyword_detected_state = State()
        space_after_projected_attributes.add_transition(
            Transition('f', from_keyword_detected_state)
        )
        from_keyword_read_state = self.__expect_phrase(
            from_keyword_detected_state,
            'rom'
        )
        space_after_from = State()
        from_keyword_read_state.add_transition(
            Transition(' ', space_after_from)
        )
        space_after_from.add_transition(
            Transition(' ', space_after_from)
        )
        relation_name_reading_state = RelationNameEditingState()
        space_after_from.add_transition(
            Transition(' ', relation_name_reading_state,
                       anything_but_transition=True)
        )
        relation_name_reading_state.add_transition(
            Transition(' ', relation_name_reading_state,
                       anything_but_transition=True)
        )
        relation_name_taken_state = State()
        relation_name_reading_state.add_transition(
            Transition(' ', relation_name_taken_state)
        )
        relation_name_taken_state.add_transition(
            Transition(' ', relation_name_taken_state)
        )
        reading_as_after_relation_name = State()
        relation_name_taken_state.add_transition(
            Transition('a', reading_as_after_relation_name)
        )
        as_after_relation_name_read = State()
        reading_as_after_relation_name.add_transition(
            Transition('s', as_after_relation_name_read)
        )
        ready_for_relation_alias_state = State()
        as_after_relation_name_read.add_transition(
            Transition(' ', ready_for_relation_alias_state)
        )
        ready_for_relation_alias_state.add_transition(
            Transition(' ', ready_for_relation_alias_state)
        )
        relation_alias_editing_state = RelationAliasEditingState()
        ready_for_relation_alias_state.add_transition(
            Transition(' ', relation_alias_editing_state,
                       anything_but_transition=True)
        )
        relation_alias_editing_state.add_transition(
            Transition(' ', relation_alias_editing_state,
                       anything_but_transition=True)
        )
        relation_alias_editing_state.add_transition(
            Transition(' ', relation_name_taken_state)
        )
        repeat_keyword_detected_state = AddRepeatConstraintState()
        relation_name_taken_state.add_transition(
            Transition('r', repeat_keyword_detected_state)
        )
        repeat_keyword_almost_read_state = self.__expect_phrase(
            repeat_keyword_detected_state, 'epeat')
        repeat_keyword_read_state = State()
        repeat_keyword_almost_read_state.add_transition(
            Transition(' ', repeat_keyword_read_state)
        )
        repeat_keyword_read_state.add_transition(
            Transition(' ', repeat_keyword_read_state)
        )
        repetition_limit_editing_state = RepetitionLimitEditingState()
        for digit in self.__digits:
            repeat_keyword_read_state.add_transition(
                Transition(digit, repetition_limit_editing_state)
            )
            repetition_limit_editing_state.add_transition(
                Transition(digit, repetition_limit_editing_state)
            )
        repetition_limit_editing_state.add_transition(
            Transition(' ', relation_name_taken_state)
        )
        where_keyword_detected_state = State()
        relation_name_taken_state.add_transition(
            Transition('w', where_keyword_detected_state)
        )
        where_read_state = self.__expect_phrase(
            where_keyword_detected_state, 'here')
        ready_for_base_predicate = State()
        where_read_state.add_transition(
            Transition(' ', ready_for_base_predicate)
        )
        ready_for_base_predicate.add_transition(
            Transition(' ', ready_for_base_predicate)
        )
        base_predicate_edition_state = BasePredicateEditingState()
        ready_for_base_predicate.add_transition(
            Transition(' ', base_predicate_edition_state,
                       anything_but_transition=True)
        )
        base_predicate_edition_state.add_transition(
            Transition('#', base_predicate_edition_state,
                       anything_but_transition=True)
        )
        such_that_found_state = State()
        base_predicate_edition_state.add_transition(
            Transition('#', such_that_found_state)
        )
        relation_name_taken_state.add_transition(
            Transition('#', such_that_found_state)
        )
        ready_for_constraints_state = self.__expect_phrase(
            such_that_found_state, 'such that ')
        ready_for_constraints_state.add_transition(
            Transition(' ', ready_for_constraints_state)
        )
        package_size_constraint_detected = AddPackageSizeConstraintState()
        ready_for_constraints_state.add_transition(
            Transition('c', package_size_constraint_detected)
        )
        count_keyword_read_state = self.__expect_phrase(
            package_size_constraint_detected, 'ount(*)')
        count_keyword_read_state.add_transition(
            Transition(' ', count_keyword_read_state)
        )
        inequality_sign_read_state = ConstraintInequalitySettingState()
        count_keyword_read_state.add_transition(
            Transition('>', inequality_sign_read_state)
        )
        count_keyword_read_state.add_transition(
            Transition('<', inequality_sign_read_state)
        )
        equality_sign_read_state = ConstraintInequalitySettingState()
        count_keyword_read_state.add_transition(
            Transition('=', equality_sign_read_state)
        )
        inequality_sign_completed_state = State()
        inequality_sign_read_state.add_transition(
            Transition('=', inequality_sign_completed_state)
        )
        space_after_inequality_sign = State()
        equality_sign_read_state.add_transition(
            Transition(' ', space_after_inequality_sign)
        )
        inequality_sign_completed_state.add_transition(
            Transition(' ', space_after_inequality_sign)
        )
        space_after_inequality_sign.add_transition(
            Transition(' ', space_after_inequality_sign)
        )
        package_size_limit_editing_state = \
            PackageSizeLimitEditingState()
        for digit in self.__digits:
            inequality_sign_completed_state.add_transition(
                Transition(digit, package_size_limit_editing_state)
            )
            equality_sign_read_state.add_transition(
                Transition(digit, package_size_limit_editing_state)
            )
            space_after_inequality_sign.add_transition(
                Transition(digit, package_size_limit_editing_state)
            )
            package_size_limit_editing_state.add_transition(
                Transition(digit, package_size_limit_editing_state)
            )
        package_size_limit_taken_state = State()
        package_size_limit_editing_state.add_transition(
            Transition(' ', package_size_limit_taken_state)
        )
        package_size_limit_taken_state.add_transition(
            Transition(' ', package_size_limit_taken_state)
        )
        ready_for_next_constraint = self.__expect_phrase(
            package_size_limit_taken_state, 'and')
        ready_for_next_constraint.add_transition(
            Transition(' ', ready_for_constraints_state)
        )
        ready_for_objective_state = State()
        package_size_limit_taken_state.add_transition(
            Transition('m', ready_for_objective_state)
        )
        set_min_objective_state = ObjectiveTypeState()
        set_max_objective_state = ObjectiveTypeState()
        ready_for_objective_state.add_transition(
            Transition('i', set_min_objective_state)
        )
        ready_for_objective_state.add_transition(
            Transition('a', set_max_objective_state)
        )
        maximize_read_state = self.__expect_phrase(
            set_max_objective_state, 'ximize'
        )
        minimize_read_state = self.__expect_phrase(
            set_min_objective_state, 'nimize'
        )
        objective_type_read_state = State()
        minimize_read_state.add_transition(
            Transition(' ', objective_type_read_state)
        )
        maximize_read_state.add_transition(
            Transition(' ', objective_type_read_state)
        )
        objective_type_read_state.add_transition(
            Transition(' ', objective_type_read_state)
        )
        expected_sum_detected_state = \
            ObjectiveStochasticityState()
        sum_detected_state = \
            ObjectiveStochasticityState()
        objective_type_read_state.add_transition(
            Transition('e', expected_sum_detected_state)
        )
        objective_type_read_state.add_transition(
            Transition('s', sum_detected_state)
        )
        expected_sum_read_state = self.__expect_phrase(
            expected_sum_detected_state, 'xpected sum'
        )
        sum_read_state = self.__expect_phrase(
            sum_detected_state, 'um'
        )
        ready_for_objective_attribute_state = State()
        expected_sum_read_state.add_transition(
            Transition('(', ready_for_objective_attribute_state)
        )
        sum_read_state.add_transition(
            Transition('(', ready_for_objective_attribute_state)
        )
        objective_attribute_editing_state = \
            ObjectiveAttributeNameEditingState()
        ready_for_objective_attribute_state.add_transition(
            Transition(')', objective_attribute_editing_state,
                       anything_but_transition=True)
        )
        objective_attribute_editing_state.add_transition(
            Transition(')', objective_attribute_editing_state,
                       anything_but_transition=True)
        )
        self.__query_successfully_parsed_state = State()
        objective_attribute_editing_state.add_transition(
            Transition(')', self.__query_successfully_parsed_state)
        )
        add_deterministic_constraint_state = \
            AddDeterministicConstraintState()
        ready_for_constraints_state.add_transition(
            Transition('s', add_deterministic_constraint_state)
        )
        ready_for_deterministic_constraint_attribute = \
            self.__expect_phrase(add_deterministic_constraint_state,
                               'um(')
        deterministic_constraint_attribute_editing_state = \
            ConstraintAttributeNameEditingState()
        ready_for_deterministic_constraint_attribute.add_transition(
            Transition(')', deterministic_constraint_attribute_editing_state,
                       anything_but_transition=True)
        )
        deterministic_constraint_attribute_editing_state.add_transition(
            Transition(')', deterministic_constraint_attribute_editing_state,
                       anything_but_transition=True)
        )
        constraint_attribute_read_state = State()
        deterministic_constraint_attribute_editing_state.add_transition(
            Transition(')', constraint_attribute_read_state)
        )
        deterministic_inequality_read_state = \
            ConstraintInequalitySettingState()
        constraint_attribute_read_state.add_transition(
            Transition(' ', constraint_attribute_read_state)
        )
        constraint_attribute_read_state.add_transition(
            Transition('<', deterministic_inequality_read_state)
        )
        constraint_attribute_read_state.add_transition(
            Transition('>', deterministic_inequality_read_state)
        )
        ready_for_sum_limit_state = State()
        deterministic_inequality_read_state.add_transition(
            Transition('=', ready_for_sum_limit_state)
        )
        deterministic_equality_read_state = \
            ConstraintInequalitySettingState()
        constraint_attribute_read_state.add_transition(
            Transition('=', deterministic_equality_read_state)
        )
        constraint_space_after_relational_operator_state = State()
        deterministic_equality_read_state.add_transition(
            Transition(' ', 
                       constraint_space_after_relational_operator_state)
        )
        ready_for_sum_limit_state.add_transition(
            Transition(' ', 
                       constraint_space_after_relational_operator_state)
        )
        constraint_space_after_relational_operator_state.add_transition(
            Transition(' ',
                       constraint_space_after_relational_operator_state)
        )
        constraint_sum_limit_editing_state = \
            ConstraintSumLimitSettingState()
        for numerical in self.__numericals:
            constraint_sum_limit_editing_state.add_transition(
                Transition(numerical, constraint_sum_limit_editing_state)
            )
            deterministic_equality_read_state.add_transition(
                Transition(numerical, constraint_sum_limit_editing_state)
            )
            ready_for_sum_limit_state.add_transition(
                Transition(numerical, constraint_sum_limit_editing_state)
            )
            constraint_space_after_relational_operator_state.add_transition(
                Transition(numerical, constraint_sum_limit_editing_state)
            )
        constraint_sum_limit_parsed_state = State()
        constraint_sum_limit_editing_state.add_transition(
            Transition(' ', constraint_sum_limit_parsed_state)
        )
        var_constraint_detected_state = TurnToVaRConstraintState()
        constraint_sum_limit_parsed_state.add_transition(
            Transition('w', var_constraint_detected_state)
        )
        with_probability_read_state = self.__expect_phrase(
            var_constraint_detected_state, 'ith probability'
        )
        optional_space_state = State()
        with_probability_read_state.add_transition(
            Transition(' ', optional_space_state)
        )
        optional_space_state.add_transition(
            Transition(' ', optional_space_state)
        )
        interim_state = State()
        with_probability_read_state.add_transition(
            Transition('>', interim_state)
        )
        optional_space_state.add_transition(
            Transition('>', interim_state)
        )
        with_probability_and_inequality_read_state = State()
        interim_state.add_transition(
            Transition('=', with_probability_and_inequality_read_state)
        )
        with_probability_and_inequality_read_state.add_transition(
            Transition(' ', with_probability_and_inequality_read_state)
        )
        probability_threshold_editing_state = \
            ProbabilityThresholdEditingState()
        for numerical in self.__numericals:
            with_probability_and_inequality_read_state.add_transition(
                Transition(numerical, probability_threshold_editing_state)
            )
            probability_threshold_editing_state.add_transition(
                Transition(numerical, probability_threshold_editing_state)
            )
        probability_threshold_editing_state.add_transition(
            Transition(' ', constraint_sum_limit_parsed_state)
        )
        constraint_sum_limit_parsed_state.add_transition(
            Transition(' ', constraint_sum_limit_parsed_state)
        )
        constraint_sum_limit_parsed_state.add_transition(
            Transition('m', ready_for_objective_state)
        )
        reading_a_in_and_state = State()
        reading_n_in_and_state = State()
        constraint_sum_limit_parsed_state.add_transition(
            Transition('a', reading_a_in_and_state)
        )
        reading_a_in_and_state.add_transition(
            Transition('n', reading_n_in_and_state)
        )
        reading_n_in_and_state.add_transition(
            Transition('d', ready_for_next_constraint)
        )
        add_expected_value_constraint = State()
        ready_for_constraints_state.add_transition(
            Transition('e', add_expected_value_constraint)
        )
        ready_for_expected_sum_attribute = self.__expect_phrase(
            add_expected_value_constraint, 'xpected sum(')
        expected_sum_attribute_editing_state = ConstraintAttributeNameEditingState()
        ready_for_expected_sum_attribute.add_transition(
            Transition(')', expected_sum_attribute_editing_state,
                       anything_but_transition=True)
        )
        expected_sum_attribute_editing_state.add_transition(
            Transition(')', expected_sum_attribute_editing_state,
                       anything_but_transition=True)
        )
        expected_sum_attribute_read_state = State()
        expected_sum_attribute_editing_state.add_transition(
            Transition(')', expected_sum_attribute_read_state)
        )
        expected_sum_inequality_sign_started_state = ConstraintInequalitySettingState()
        expected_sum_attribute_read_state.add_transition(
            Transition('>', expected_sum_inequality_sign_started_state)
        )
        expected_sum_attribute_read_state.add_transition(
            Transition('<', expected_sum_inequality_sign_started_state)
        )
        expected_sum_inequality_sign_read_state = State()
        expected_sum_inequality_sign_started_state.add_transition(
            Transition('=', expected_sum_inequality_sign_read_state)
        )
        expected_sum_equality_sign_read_state = ConstraintInequalitySettingState()
        expected_sum_attribute_read_state.add_transition(
            Transition('=', expected_sum_equality_sign_read_state)
        )
        expected_sum_limit_editing_state = ConstraintSumLimitSettingState()
        for numerical in self.__numericals:
            expected_sum_equality_sign_read_state.add_transition(
                Transition(numerical, expected_sum_limit_editing_state)
            )
            expected_sum_inequality_sign_read_state.add_transition(
                Transition(numerical, expected_sum_limit_editing_state)
            )
            expected_sum_limit_editing_state.add_transition(
                Transition(numerical, expected_sum_limit_editing_state)
            )
        expected_sum_limit_parsed_state = State()
        expected_sum_limit_editing_state.add_transition(
            Transition(' ', expected_sum_limit_parsed_state)
        )
        expected_sum_limit_parsed_state.add_transition(
            Transition(' ', expected_sum_limit_parsed_state)
        )
        expected_sum_limit_parsed_state.add_transition(
            Transition('a', reading_a_in_and_state)
        )
        expected_sum_limit_parsed_state.add_transition(
            Transition('m', ready_for_objective_state)
        )
    
    def __collapse_into_one_line(self, query_lines) -> str:
        query = ''
        for line in query_lines:
            query += ' '
            query += line
        return query
    
    def __insert_special_character_before_such_that(self, query: str) -> str:
        position = query.find('such that')
        if position == -1:
            raise Exception
        return query[0:position] + '#' + query[position:]
    
    def preprocess(self, query_lines) -> str:
        query = self.__collapse_into_one_line(query_lines)
        query = query.lower()
        query = self.__insert_special_character_before_such_that(query)
        return query
    
    def parse(self, query_lines) -> Query:
        query_str = self.preprocess(query_lines)
        state = self.__init_state
        query = Query()
        for character in query_str:
            state = state.get_next_state(character)
            query = state.process(query, character)
        if state != self.__query_successfully_parsed_state:
            raise Exception
        return query


