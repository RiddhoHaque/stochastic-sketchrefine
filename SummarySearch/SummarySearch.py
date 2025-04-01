import gurobipy as gp
from gurobipy import GRB
import numpy as np
import psutil
from scipy import optimize

from DbInfo.DbInfo import DbInfo
from OptimizationMetrics.OptimizationMetrics import OptimizationMetrics
from PgConnection.PgConnection import PgConnection
from SeedManager.SeedManager import SeedManager
from StochasticPackageQuery.Constraints.VaRConstraint.VaRConstraint import VaRConstraint
from StochasticPackageQuery.Query import Query
from Utils.ArcTangent import ArcTangent
from Utils.GurobiLicense import GurobiLicense
from Utils.ObjectiveType import ObjectiveType
from Utils.RelationalOperators import RelationalOperators
from Utils.Stochasticity import Stochasticity
from Validator.Validator import Validator
from ValueGenerator.ValueGenerator import ValueGenerator


class SummarySearch:

    def __init__(self, query: Query,
                 linear_relaxation: bool,
                 dbInfo: DbInfo,
                 init_no_of_scenarios: int,
                 init_no_of_summaries: int,
                 no_of_validation_scenarios: int,
                 approximation_bound: float):
        self.__query = query
        self.__gurobi_env = gp.Env(
            params=GurobiLicense.OPTIONS)
        self.__gurobi_env.setParam(
            'OutputFlag', 0
        )
        self.__model = gp.Model(
            env=self.__gurobi_env)
        self.__is_linear_relaxation = \
            linear_relaxation
        self.__init_no_of_scenarios = \
            init_no_of_scenarios
        self.__init_no_of_summaries = \
            init_no_of_summaries

        self.__no_of_validation_scenarios = \
            no_of_validation_scenarios
        self.__validator = Validator(
            self.__query, dbInfo,
            no_of_validation_scenarios
        )
        self.__approximation_bound = \
            approximation_bound

        self.__no_of_vars = \
            self.__get_number_of_tuples()
        self.__feasible_no_of_scenarios_to_store = \
            int(np.floor(
                (0.40*psutil.virtual_memory().available)/\
                (64*self.__no_of_vars)))
        
        self.__vars = []

        self.__current_number_of_scenarios = 0
        self.__scenarios = dict()
        for attr in self.__get_stochastic_attributes():
            self.__scenarios[attr] = []
            for _ in range(self.__no_of_vars):
                self.__scenarios[attr].append([])
        
        self.__values = dict()
        for attr in self.__get_deterministic_attributes():
            values = \
                ValueGenerator(
                    relation=self.__query.get_relation(),
                    base_predicate=self.__query.get_base_predicate(),
                    attribute=attr).get_values()
            self.__values[attr] = []
            for value in values:
                self.__values[attr].append(value[0])

        self.__dbInfo = dbInfo
        self.__ids = []
        ids = ValueGenerator(
                relation=self.__query.get_relation(),
                base_predicate=self.__query.get_base_predicate(),
                attribute='id'
            ).get_values()
        for id in ids:
            self.__ids.append(id[0])
        
        self.__metrics = OptimizationMetrics(
            'SummarySearch', self.__is_linear_relaxation)


    def __get_stochastic_attributes(self):
        attributes = set()
        for constraint in self.__query.get_constraints():
            if constraint.is_expected_sum_constraint():
                attributes.add(
                    constraint.get_attribute_name())
            if constraint.is_risk_constraint():
                attributes.add(
                    constraint.get_attribute_name())
        if self.__query.get_objective().get_stochasticity() \
            == Stochasticity.STOCHASTIC:
            attributes.add(
                self.__query.\
                    get_objective().\
                        get_attribute_name()
            )
        return attributes

    
    def __get_deterministic_attributes(self):
        attributes = set()
        for constraint in self.__query.get_constraints():
            if constraint.is_deterministic_constraint():
                attributes.add(
                    constraint.get_attribute_name())
        
        if self.__query.get_objective().get_stochasticity() \
            == Stochasticity.DETERMINISTIC:
            attributes.add(
                self.__query.\
                    get_objective().\
                        get_attribute_name()
            )
        return attributes
    
    
    def __get_number_of_tuples(self):
        sql_query = "SELECT COUNT(*) FROM " \
            + self.__query.get_relation()
        
        if len(self.__query.get_base_predicate()) > 0:
            sql_query += " WHERE "+ \
                self.__query.get_base_predicate()
        
        sql_query += ";"
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()[0][0]

    
    def __get_upper_bound_for_vars(self):
        for constraint in \
            self.__query.get_constraints():
            if constraint.is_repeat_constraint():
                return 1 + constraint.get_repetition_limit()
        return None
    
    
    def __add_package_size_constraint_to_model(
            self, package_size_constraint):
        size_limit = \
            package_size_constraint.get_package_size_limit()
        inequality_sign = \
            package_size_constraint.get_inequality_sign()
        
        gurobi_inequality = GRB.LESS_EQUAL
        if inequality_sign == RelationalOperators.EQUALS:
            gurobi_inequality = GRB.EQUAL
        elif inequality_sign == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            gurobi_inequality = GRB.GREATER_EQUAL
        
        self.__model.addConstr(
            gp.LinExpr([1]*self.__no_of_vars, self.__vars),
            gurobi_inequality, size_limit
        )

    
    def __add_deterministic_constraint_to_model(
        self, deterministic_constraint):
        attribute = deterministic_constraint.get_attribute_name()

        inequality_sign = \
            deterministic_constraint.get_inequality_sign()
        gurobi_inequality = GRB.LESS_EQUAL
        if inequality_sign == RelationalOperators.EQUALS:
            gurobi_inequality = GRB.EQUAL
        elif inequality_sign == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            gurobi_inequality = GRB.GREATER_EQUAL
        
        sum_limit = deterministic_constraint.get_sum_limit()
        self.__model.addConstr(
            gp.LinExpr(self.__values[attribute], self.__vars),
            gurobi_inequality, sum_limit
        )

    
    def __add_expected_sum_constraint_to_model(
        self, expected_sum_constraint,
        no_of_scenarios):
        attr = expected_sum_constraint.get_attribute_name()
        coefficients = []
        
        if no_of_scenarios <= \
            self.__feasible_no_of_scenarios_to_store:
            for idx in range(self.__no_of_vars):
                coefficients.append(
                    np.average(
                        self.__scenarios[attr][idx]
                    )
                )
        
        else:
            total_scenarios = 0
            coefficient_set = [[] for _ in range(
                self.__no_of_vars)]
            while total_scenarios < no_of_scenarios:
                self.__add_feasible_no_of_scenarios(attr)
                total_scenarios += \
                    self.__feasible_no_of_scenarios_to_store
                for idx in range(self.__no_of_vars):
                    coefficient_set[idx].append(
                        np.average(
                            self.__scenarios[attr][idx]
                        )
                    )
            for idx in range(self.__no_of_vars):
                coefficients.append(
                    np.average(coefficient_set[idx])
                )

        inequality_sign = \
            expected_sum_constraint.get_inequality_sign()
        gurobi_inequality = GRB.LESS_EQUAL
        if inequality_sign == RelationalOperators.EQUALS:
            gurobi_inequality = GRB.EQUAL
        elif inequality_sign == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            gurobi_inequality = GRB.GREATER_EQUAL
        
        sum_limit = expected_sum_constraint.get_sum_limit()

        self.__model.addConstr(
            gp.LinExpr(coefficients, self.__vars),
            gurobi_inequality, sum_limit
        )
    

    def __add_summaries_to_model(
        self, var_constraint,
        no_of_scenarios,
        no_of_summaries,
        alpha, previous_package
    ):
        indicators = []
        scenarios = [_ for _ in range(no_of_scenarios)]
        np.random.shuffle(scenarios)
        scenario_index = 0
        
        for summary_index in range(int(no_of_summaries)):
            scenarios_per_summary = \
                self.__get_scenarios_for_this_summary(
                    alpha, no_of_scenarios,
                    no_of_summaries, summary_index
                )
            # print('scenarios for this summary:', scenarios_per_summary)
            indicators.append(
                self.__create_summary(
                    scenarios[
                        scenario_index:
                        scenario_index + \
                            scenarios_per_summary],
                    var_constraint,
                    alpha, previous_package
                )
            )
            scenario_index += scenarios_per_summary

        return indicators

    
    def __add_var_constraint_to_model(
        self, var_constraint,
        no_of_scenarios,
        no_of_summaries,
        alpha = None,
        previous_package = None,
    ):
        indicators = []

        if no_of_scenarios > self.__feasible_no_of_scenarios_to_store:
            total_scenarios = 0
            number_of_fragments = np.ceil(
                no_of_scenarios / self.__feasible_no_of_scenarios_to_store)
            summaries_per_fragment = np.floor(
                no_of_summaries / number_of_fragments)
            fragment_number = 0
            
            while total_scenarios < no_of_scenarios:
                self.__add_feasible_no_of_scenarios(
                    var_constraint.get_attribute_name())
                
                summaries_for_this_fragment = summaries_per_fragment
                if fragment_number < \
                    no_of_summaries % number_of_fragments:
                    summaries_for_this_fragment += 1
                if summaries_for_this_fragment == 0:
                    summaries_for_this_fragment = 1
                
                fragment_indicators = self.__add_summaries_to_model(
                    var_constraint, self.__feasible_no_of_scenarios_to_store,
                    summaries_for_this_fragment, alpha, previous_package
                )
                for indicator in fragment_indicators:
                    indicators.append(indicator)

                fragment_number += 1
                total_scenarios += self.__feasible_no_of_scenarios_to_store        
        else:
            indicators = self.__add_summaries_to_model(
                var_constraint, no_of_scenarios,
                no_of_summaries, alpha, previous_package
            )
        # print('Sum of indicators >=', len(indicators) * var_constraint.get_probability_threshold())
        self.__model.addConstr(
            gp.LinExpr([1]*len(indicators), indicators),
            GRB.GREATER_EQUAL,
            len(indicators) * var_constraint.get_probability_threshold()
        )


    def __create_summary(self, scenarios: list[int],
                         var_constraint: VaRConstraint,
                         alpha: float,
                         previous_package):
        package_tuple_indices = dict()
        index = 0
        if previous_package is not None:
            for id in self.__ids:
                if id in previous_package:
                    package_tuple_indices[index] = \
                        previous_package[id]
                index += 1
        
        scenarios_with_scores = []
        for scenario_no in scenarios:
            scenario_score = 0
            for index in package_tuple_indices:
                scenario_score += self.__scenarios[
                    var_constraint.get_attribute_name()][
                    index][scenario_no] * \
                    package_tuple_indices[index]
            scenarios_with_scores.append(
                (scenario_score, scenario_no)
            )
        
        if var_constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            scenarios_with_scores.sort(reverse=True)
        else:
            scenarios_with_scores.sort()
        
        scenarios_to_consider = int(np.ceil(
            alpha * len(scenarios)))
        # print('Scenarios to consider', scenarios_to_consider)
        
        no_of_scenarios_considered = 0
        coefficients = []
        for _ , scenario_no in scenarios_with_scores:
            if no_of_scenarios_considered == 0:
                for var in range(self.__no_of_vars):
                    coefficients.append(
                        self.__scenarios[
                            var_constraint.get_attribute_name()
                        ][var][scenario_no]
                    )
            elif var_constraint.get_inequality_sign() == \
                RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                for var in range(self.__no_of_vars):
                    if self.__scenarios[
                        var_constraint.get_attribute_name()
                        ][var][scenario_no] < \
                        coefficients[var]:
                        coefficients[var] = \
                            self.__scenarios[
                            var_constraint.get_attribute_name()
                            ][var][scenario_no]
            else:
                for var in range(self.__no_of_vars):
                    if self.__scenarios[
                        var_constraint.get_attribute_name()
                        ][var][scenario_no] > \
                        coefficients[var]:
                        coefficients[var] = \
                            self.__scenarios[
                            var_constraint.get_attribute_name()
                            ][var][scenario_no]
            
            no_of_scenarios_considered += 1
            if no_of_scenarios_considered == scenarios_to_consider:
                break
        
        indicator = self.__model.addVar(vtype=GRB.BINARY)
        
        gurobi_inequality = GRB.LESS_EQUAL
        if var_constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            gurobi_inequality = GRB.GREATER_EQUAL

        self.__model.addGenConstrIndicator(
            indicator, True, gp.LinExpr(coefficients, self.__vars),
            gurobi_inequality, var_constraint.get_sum_limit()
        )

        return indicator

    
    def __add_feasible_no_of_scenarios(self, attribute):
        self.__scenarios[attribute] = \
            self.__dbInfo.get_variable_generator_function(
                attribute)(
                    self.__query.get_relation(),
                    self.__query.get_base_predicate()
                ).generate_scenarios(
                    seed = SeedManager.get_next_seed(),
                    no_of_scenarios = \
                        self.__feasible_no_of_scenarios_to_store
                )
    
    
    def __add_scenarios_if_necessary(self, no_of_scenarios):
        if no_of_scenarios > self.__feasible_no_of_scenarios_to_store:
            return

        if self.__current_number_of_scenarios < \
            no_of_scenarios:
            for attr in self.__scenarios:
                new_scenarios = \
                    self.__dbInfo.get_variable_generator_function(
                        attr)(
                            self.__query.get_relation(),
                            self.__query.get_base_predicate()
                        ).generate_scenarios(
                            seed = SeedManager.get_next_seed(),
                            no_of_scenarios = no_of_scenarios - \
                                self.__current_number_of_scenarios
                        )
                for ind in range(len(new_scenarios)):
                    for value in new_scenarios[ind]:
                        self.__scenarios[attr][ind].append(value)
            self.__current_number_of_scenarios = no_of_scenarios


    def __add_constraints_to_model(
        self, no_of_scenarios, probabilistically_constrained,
        no_of_summaries, alpha, previous_package):
        if probabilistically_constrained:
            self.__add_scenarios_if_necessary(no_of_scenarios)

        var_constraint_index = 0
        for constraint in self.__query.get_constraints():
            if constraint.is_package_size_constraint():
                self.__add_package_size_constraint_to_model(
                    constraint
                )
            if constraint.is_deterministic_constraint():
                self.__add_deterministic_constraint_to_model(
                    constraint
                )
            if constraint.is_expected_sum_constraint():
                if probabilistically_constrained:
                    self.__add_expected_sum_constraint_to_model(
                        constraint, no_of_scenarios
                    )
            if constraint.is_var_constraint():
                if probabilistically_constrained:
                    if alpha[var_constraint_index] > 0:
                        self.__add_var_constraint_to_model(
                            constraint, no_of_scenarios,
                            no_of_summaries,
                            alpha[var_constraint_index],
                            previous_package
                        )
                    var_constraint_index += 1 

    
    def __add_objective_to_model(
        self, objective, no_of_scenarios):
        attr = objective.get_attribute_name()
        coefficients = []
        
        if objective.get_stochasticity() == \
            Stochasticity.DETERMINISTIC:
            coefficients = self.__values[attr]
        else:
            if no_of_scenarios <= \
                self.__feasible_no_of_scenarios_to_store:
                self.__add_scenarios_if_necessary(
                    no_of_scenarios)
                for idx in range(self.__no_of_vars):
                    coefficients.append(
                        np.average(
                            self.__scenarios[attr][idx]
                        )
                    )
            else:
                total_scenarios = 0
                coefficient_set = [[] for _ in \
                                   range(self.__no_of_vars)]
                while total_scenarios < no_of_scenarios:
                    self.__add_feasible_no_of_scenarios(attr)
                    total_scenarios += \
                        self.__feasible_no_of_scenarios_to_store
                    for idx in range(self.__no_of_vars):
                        coefficient_set[idx].append(
                            np.average(
                                self.__scenarios[attr][idx]
                            )
                        )
                for idx in range(self.__no_of_vars):
                    coefficients.append(
                        np.average(coefficient_set[idx])
                    )
        
        objective_type = objective.get_objective_type()
        
        gurobi_objective = GRB.MAXIMIZE
        if objective_type == ObjectiveType.MINIMIZATION:
            gurobi_objective = GRB.MINIMIZE
        
        self.__model.setObjective(
            gp.LinExpr(coefficients, self.__vars),
            gurobi_objective)
                
    
    def __add_variables_to_model(self):
        max_repetition = \
            self.__get_upper_bound_for_vars()
        type = GRB.INTEGER
        if self.__is_linear_relaxation:
            type = GRB.CONTINUOUS
        self.__vars = []
        if max_repetition is not None:
            for _ in range(self.__no_of_vars):
                self.__vars.append(
                    self.__model.addVar(
                        lb=0,
                        ub=max_repetition,
                        vtype=type
                    )
                )
        else:
            for _ in range(self.__no_of_vars):
                self.__vars.append(
                    self.__model.addVar(
                        lb = 0,
                        vtype = type
                    )
                )
    
    
    def __model_setup(
            self, no_of_scenarios: int,
            probabilistically_constrained = True,
            no_of_summaries = 0,
            alpha = None,
            previous_package = None,):
        
        self.__model = gp.Model(
            env=self.__gurobi_env)
        self.__add_variables_to_model()
        self.__add_constraints_to_model(
            no_of_scenarios, probabilistically_constrained,
            no_of_summaries, alpha, previous_package)
        self.__add_objective_to_model(
            self.__query.get_objective(), no_of_scenarios)
    
    
    def __get_package(self):
        self.__metrics.start_optimizer()
        self.__model.optimize()
        self.__metrics.end_optimizer()
        package_dict = {}
        idx = 0
        try:
            for var in self.__vars:
                if var.x > 0:
                    package_dict[self.__ids[idx]] = var.x
                idx += 1
        except AttributeError:
            return None
        return package_dict


    def __get_scenarios_for_this_summary(
        self, alpha, no_of_scenarios,
        no_of_summaries, summary_index
    ):
        scenarios_per_summary = \
            int(np.floor(
                no_of_scenarios / no_of_summaries))
        if summary_index < \
            (no_of_scenarios % no_of_summaries):
            scenarios_per_summary += 1
        return int(np.ceil(
            alpha*scenarios_per_summary))


    def __get_no_of_scenarios_considered(
        self, alpha, no_of_scenarios,
        no_of_summaries, summary_index):
        scenarios_per_summary = \
            self.__get_scenarios_for_this_summary(
                alpha, no_of_scenarios,
                no_of_summaries, summary_index
            )
        return int(np.ceil(
            alpha*scenarios_per_summary))

    
    def __csa_solve(self, no_of_scenarios,
                    no_of_summaries, alpha_histories,
                    previous_package, upper_bound):
        alphas = []
        history = []
        clean_history = []
        for alpha_history in alpha_histories:
            if len(alpha_history) > 1:
                alpha_history = alpha_history[:1]
            clean_history.append(alpha_history)
        alpha_histories = clean_history
        # print('Initial alpha history:', alpha_histories)
        best_feasible_package = previous_package
        best_objective_value = 0
        alpha_histories_copy = alpha_histories

        while True:
            alphas = []
            no_of_scenarios_considered = []
            var_constraint_index = 0
            for constraint in self.__query.get_constraints():
                if constraint.is_var_constraint():
                    alpha_history = \
                        alpha_histories_copy[
                            var_constraint_index]
                    alpha_data = []
                    y_data = []
                    # print('Alpha history', alpha_history)
                    for alpha, y in alpha_history:
                        alpha_data.append(alpha)
                        y_data.append(y)
                        if alpha == 1 and y < 0:
                            return (best_feasible_package, upper_bound)
                    if len(alpha_data) == 1:
                        next_alpha = 1
                    else:
                        # print('Alpha data:', alpha_data)
                        # print('y data:', y_data)
                        params, _ = optimize.curve_fit(
                            ArcTangent.func, alpha_data,
                            y_data)
                        next_alpha = np.tan(-params[1]/params[0])
                    # print('Next alpha', next_alpha)
                    alphas.append(next_alpha)
                    no_of_scenarios_considered.append(
                        self.__get_no_of_scenarios_considered(
                            next_alpha, no_of_scenarios,
                            no_of_summaries, var_constraint_index
                        )
                    )
                    #print('No of scenarios considered',
                    #      no_of_scenarios_considered[-1])
                    var_constraint_index += 1
            if (no_of_scenarios_considered, previous_package) in history:
                return (best_feasible_package, upper_bound)
            
            history.append((no_of_scenarios_considered, previous_package))
            
            self.__model_setup(
                no_of_scenarios=no_of_scenarios,
                probabilistically_constrained=True,
                no_of_summaries=no_of_summaries,
                alpha=alphas,
                previous_package=previous_package
            )
            previous_package = self.__get_package()
            
            if self.__validator.is_package_validation_feasible(
                previous_package):
                
                if self.__validator.is_package_1_pm_epsilon_approximate(
                    previous_package, self.__approximation_bound,
                    upper_bound):
                        return (previous_package, upper_bound)
                else:
                    objective_value = \
                        self.__validator.get_validation_objective_value(
                            previous_package)
                    if self.__query.get_objective().get_objective_type() == \
                        ObjectiveType.MAXIMIZATION:
                        if objective_value > best_objective_value:
                            best_feasible_package = previous_package
                            best_objective_value = objective_value
                    elif objective_value < best_objective_value:
                        best_feasible_package = previous_package
                        best_objective_value = objective_value
            
            else:
                objective_value = self.__validator.get_validation_objective_value(
                    previous_package
                )
                if self.__query.get_objective().get_objective_type() == \
                    ObjectiveType.MAXIMIZATION:
                    if objective_value < upper_bound:
                        ...
                        # upper_bound = objective_value
                elif objective_value > upper_bound:
                    ...
                    # upper_bound = objective_value

            var_constraint_index = 0
            for constraint in self.__query.get_constraints():
                if constraint.is_var_constraint():
                    alpha_histories_copy[var_constraint_index].append(
                        (alphas[var_constraint_index],
                        self.__validator.get_var_constraint_satisfaction(
                            previous_package, constraint
                        ) - constraint.get_probability_threshold())
                    )
                    #print('Appended', alpha_histories_copy[var_constraint_index][-1])
                    var_constraint_index += 1
                
    
    def solve(self):
        self.__metrics.start_execution()
        no_of_scenarios = self.__init_no_of_scenarios
        
        self.__model_setup(
            no_of_scenarios,
            probabilistically_constrained=False)
        
        probabilistically_unconstrained_package = \
            self.__get_package()
        
        print('Probabilistically unconstrained package:',
              probabilistically_unconstrained_package)
        
        if probabilistically_unconstrained_package is None:
            self.__metrics.end_execution(
                0, 0)
            return None

        upper_bound = \
            self.__validator.get_validation_objective_value(
                package_dict=probabilistically_unconstrained_package
            )
        print('Objective Upper Bound:', upper_bound)
        alpha_histories = []

        if self.__validator.is_package_validation_feasible(
            probabilistically_unconstrained_package):
            # print('Probabilistically unconstrained package is feasible')
            # for constraint in self.__query.get_constraints():
                # if constraint.is_var_constraint():
                    #print('Passes', self.__validator.get_var_constraint_satisfaction(
                    #    package_dict=probabilistically_unconstrained_package,
                    #    var_constraint=constraint
                    #), 'of scenarios')
            self.__metrics.end_execution(upper_bound, 0)
            return (probabilistically_unconstrained_package,
                    upper_bound)

        else:
            #print('Probabilistically unconstrained package is infeasible')
            for constraint in self.__query.get_constraints():
                if constraint.is_var_constraint():
                    alpha_histories.append([])
                    alpha_histories[-1].append(
                        (0, self.__validator.get_var_constraint_satisfaction(
                                probabilistically_unconstrained_package,
                                constraint) - constraint.get_probability_threshold()
                        )
                    )
        
        no_of_summaries = self.__init_no_of_summaries
        
        while no_of_scenarios <= self.__no_of_validation_scenarios:
            #print('Trying with', no_of_scenarios, 'scenarios and',
            #      no_of_summaries, 'summaries')
            package, upper_bound = self.__csa_solve(
                no_of_scenarios=no_of_scenarios,
                no_of_summaries=no_of_summaries,
                alpha_histories=alpha_histories,
                previous_package=probabilistically_unconstrained_package,
                upper_bound=upper_bound
            )

            if self.__validator.is_package_validation_feasible(package):
                if self.__validator.is_package_1_pm_epsilon_approximate(
                    package, self.__approximation_bound, upper_bound):
                    validation_objective_value = \
                        self.__validator.get_validation_objective_value(
                            package
                        )
                    self.__metrics.end_execution(
                        validation_objective_value,
                        no_of_scenarios)
                    return (package, 
                            validation_objective_value)
                else:
                    no_of_summaries += 1
                    #print('Increasing number of summaries to',
                    #      no_of_summaries)
            
            else:
                no_of_scenarios *= 2
                #print('Increasing number of scenarios to',
                #      no_of_scenarios)
        self.__metrics.end_execution(0)
        return None


    def __get_attributes(self, id):
        sql_query = "SELECT " + \
            self.__query.get_projected_attributes() + \
            " FROM " + self.__query.get_relation() + \
            " WHERE id=" + str(id) + \
            " ORDER BY id;"
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()[0]


    def display_package(self, package_dict):
        if package_dict is None:
            return
        for id in package_dict:
            attr = self.__get_attributes(id)
            print(attr, ',', package_dict[id])

    def __del__(self):
        self.__model.close()
        self.__gurobi_env.close()

    def get_metrics(self) -> OptimizationMetrics:
        return self.__metrics