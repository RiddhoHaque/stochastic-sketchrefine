import copy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import psutil

from CVaRification.CVaRificationSearchResults import CVaRificationSearchResults
from DbInfo.DbInfo import DbInfo
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


class RCLSolve:

    def __init__(self, query: Query,
                 linear_relaxation: bool,
                 dbInfo: DbInfo,
                 init_no_of_scenarios: int,
                 no_of_validation_scenarios: int,
                 approximation_bound: float,
                 sampling_tolerance: float,
                 bisection_threshold: float,
                 max_opt_scenarios: int,
                 gurobi_env = None,
                 check_feasibility = False,
                 optimize_lcvar: bool = False,
                 early_termination: bool = False):
        self.__query = copy.deepcopy(query)
        if gurobi_env is None:
            self.__gurobi_env = gp.Env(
                params=GurobiLicense.OPTIONS)
        else:
            self.__gurobi_env = gurobi_env
        self.__gurobi_env.setParam(
            'OutputFlag', 0
        )
        self.__model = gp.Model(
            env=self.__gurobi_env)
        if check_feasibility:
            self.__model.Params.SolutionLimit = 1
            self.__model.Params.MIPFocus = 1
        self.__check_feasibility = check_feasibility
        self.__optimize_lcvar = optimize_lcvar
        self.__early_termination = early_termination
        self.__is_linear_relaxation = \
            linear_relaxation
        self.__init_no_of_scenarios = \
            init_no_of_scenarios
        
        self.__no_of_validation_scenarios = \
            no_of_validation_scenarios
        self.__validator = Validator(
            self.__query, dbInfo,
            self.__no_of_validation_scenarios
        )
        self.__approximation_bound = \
            approximation_bound
        self.__sampling_tolerance = \
            sampling_tolerance
        self.__bisection_threshold = \
            bisection_threshold
        if not check_feasibility:
            self.__max_opt_scenarios = \
                max_opt_scenarios
        else:
            self.__max_opt_scenarios = \
                init_no_of_scenarios
        self.__no_of_vars = \
            self.__get_number_of_tuples()
        self.__feasible_no_of_scenarios_to_store = \
            int(np.floor(
                (0.40*psutil.virtual_memory().available)/\
                (64*self.__no_of_vars)))

        self.__vars = []
        
        self.__risk_constraints = []
        self.__risk_to_lcvar_constraint_mapping = dict()
        self.__sorted_scenario_prefix_sums = {}
        self.__opt_scenario_scores_cache = {}

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
                    attribute=attr
                ).get_values()
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
            'RCLSolve', self.__is_linear_relaxation
        )

        if self.__query.get_objective().is_cvar_objective():
            self.__V = None
            self.__Zs = []
            # Lazy Z_s constraint state
            self.__z_groups = []          # List[frozenset[int]]
            self.__z_group_constrs = []   # List[gp.Constr]
            self.__z_lazy_active = False  # True once lazy init has run

    
    def get_validator(self):
        return self.__validator

    def __to_cvar_constraint(self, esc: ExpectedSumConstraint) -> CVaRConstraint:
        """Convert an ExpectedSumConstraint to an equivalent CVaRConstraint at 100% tail."""
        cvar = CVaRConstraint()
        cvar.set_attribute_name(esc.get_attribute_name())
        ineq = esc.get_inequality_sign()
        cvar.set_inequality_sign(
            '>' if ineq == RelationalOperators.GREATER_THAN_OR_EQUAL_TO else '<'
        )
        cvar.set_sum_limit(esc.get_sum_limit())
        cvar.set_percentage_of_scenarios(100.0)
        cvar.set_tail_type(
            'l' if ineq == RelationalOperators.GREATER_THAN_OR_EQUAL_TO else 'h'
        )
        return cvar

    def __is_searchable_stochastic_constraint(self, constraint) -> bool:
        return constraint.is_expected_sum_constraint() or \
            constraint.is_risk_constraint()

    def __get_searchable_constraints(self):
        return [
            constraint for constraint in self.__query.get_constraints()
            if self.__is_searchable_stochastic_constraint(constraint)
        ]

    def __get_stochastic_attributes(self):
        attributes = set()
        for constraint in self.__query.get_constraints():
            if constraint.is_expected_sum_constraint():
                attributes.add(
                    constraint.get_attribute_name())
            if constraint.is_risk_constraint():
                attributes.add(
                    constraint.get_attribute_name())
        if self.__query.get_objective().is_cvar_objective():
            attributes.add(
                self.__query.get_objective().\
                    get_attribute_name()
            )
        elif self.__query.get_objective().is_stochasticity_set() and \
             self.__query.get_objective().get_stochasticity() \
             == Stochasticity.STOCHASTIC:
            attributes.add(
                self.__query.get_objective().\
                    get_attribute_name()
            )
        return attributes


    def __get_deterministic_attributes(self):
        attributes = set()
        for constraint in self.__query.get_constraints():
            if constraint.is_deterministic_constraint():
                attributes.add(
                    constraint.get_attribute_name())
        
        if self.__query.get_objective().is_stochasticity_set() and \
           self.__query.get_objective().get_stochasticity() \
           == Stochasticity.DETERMINISTIC:
            attributes.add(
                self.__query.get_objective().\
                    get_attribute_name()
            )
        return attributes
    

    def __get_number_of_tuples(self):
        sql_query = "SELECT COUNT(*) FROM " \
            + self.__query.get_relation()
        
        if len(self.__query.get_base_predicate()) > 0:
            sql_query += " WHERE " + \
                self.__query.get_base_predicate()
        
        sql_query += ";"
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()[0][0]
    

    def __get_upper_bound_for_vars(self) -> int:
        for constraint in self.__query.get_constraints():
            if constraint.is_repeat_constraint():
                return 1 + constraint.get_repetition_limit()
        return None
    
    
    def __add_variables_to_model(self, no_of_scenarios: int) -> None:
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
                        lb=0,
                        vtype=type
                    )
                )

        if not self.__check_feasibility and not self.__optimize_lcvar and \
                self.__query.get_objective().is_cvar_objective():
            self.__V = self.__model.addVar(
                vtype=GRB.CONTINUOUS
            )
            self.__Zs = []
            for _ in range(no_of_scenarios):
                if self.__query.get_objective().get_tail_type() == \
                    TailType.LOWEST:
                    self.__Zs.append(
                        self.__model.addVar(
                            ub = 0,
                            vtype=GRB.CONTINUOUS
                        )
                    )
                else:
                    self.__Zs.append(
                        self.__model.addVar(
                            lb = 0,
                            vtype=GRB.CONTINUOUS
                        )
                    )

    def __get_gurobi_inequality(
            self, inequality_sign: RelationalOperators):
        if inequality_sign == RelationalOperators.EQUALS:
            return GRB.EQUAL
        if inequality_sign == RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            return GRB.GREATER_EQUAL
        return GRB.LESS_EQUAL
    

    def __add_package_size_constraint_to_model(
        self, package_size_constraint: PackageSizeConstraint
    ):
        size_limit = \
            package_size_constraint.get_package_size_limit()
        gurobi_inequality = \
            self.__get_gurobi_inequality(
                package_size_constraint.get_inequality_sign())

        self.__model.addConstr(
            gp.LinExpr([1]*self.__no_of_vars, self.__vars),
            gurobi_inequality, size_limit
        )
    

    def __add_deterministic_constraint_to_model(
        self, deterministic_constraint: DeterministicConstraint
    ):
        attribute = deterministic_constraint.get_attribute_name()
        gurobi_inequality = \
            self.__get_gurobi_inequality(
                deterministic_constraint.get_inequality_sign())
        sum_limit = deterministic_constraint.get_sum_limit()
        
        self.__model.addConstr(
            gp.LinExpr(self.__values[attribute], self.__vars),
            gurobi_inequality, sum_limit
        )


    def __add_feasible_no_of_scenarios(self, attribute: str):
        self.__scenarios[attribute] = \
            self.__dbInfo.get_variable_generator_function(
                attribute)(
                    self.__query.get_relation(),
                    base_predicate=self.__query.get_base_predicate()
                ).generate_scenarios(
                    seed = SeedManager.get_next_seed(),
                    no_of_scenarios = \
                        self.__feasible_no_of_scenarios_to_store
                )
        
    
    def __add_all_scenarios_if_possible(self, no_of_scenarios):
        if no_of_scenarios > self.__feasible_no_of_scenarios_to_store:
            return
        
        if self.__current_number_of_scenarios < \
            no_of_scenarios:
            
            for attr in self.__scenarios:
                
                new_scenarios = \
                    self.__dbInfo.get_variable_generator_function(
                        attr)(
                            self.__query.get_relation(),
                            base_predicate=self.__query.get_base_predicate()
                        ).generate_scenarios(
                            seed = SeedManager.get_next_seed(),
                            no_of_scenarios = no_of_scenarios - \
                                self.__current_number_of_scenarios
                        )
                for ind in range(len(new_scenarios)):
                    for value in new_scenarios[ind]:
                        self.__scenarios[attr][ind].append(value)
            
            self.__current_number_of_scenarios = no_of_scenarios


    def __add_expected_sum_constraint_to_model(
        self, expected_sum_constraint: ExpectedSumConstraint,
        no_of_scenarios: int,
        threshold_override: float | None = None
    ):
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
            
        gurobi_inequality = self.__get_gurobi_inequality(
            expected_sum_constraint.get_inequality_sign()
        )

        expected_sum_limit = threshold_override
        if expected_sum_limit is None:
            expected_sum_limit = expected_sum_constraint.get_sum_limit()

        return self.__model.addConstr(
            gp.LinExpr(coefficients, self.__vars),
            gurobi_inequality, expected_sum_limit
        )
    

    def __get_no_of_scenarios_to_consider(
        self, cvar_constraint: CVaRConstraint,
        no_of_scenarios: int
    ):
        return max(1, int(np.floor(
            (cvar_constraint.get_percentage_of_scenarios()\
                *no_of_scenarios)/100
        )))


    def __preprocess_lcvar_prefix_sums(self, no_of_scenarios: int) -> None:
        if no_of_scenarios > self.__feasible_no_of_scenarios_to_store:
            return
        self.__sorted_scenario_prefix_sums = {}
        for constraint in self.__query.get_constraints():
            if not constraint.is_risk_constraint():
                continue
            attr = constraint.get_attribute_name()
            if constraint.is_cvar_constraint():
                tail_type = constraint.get_tail_type()
            else:
                if constraint.get_inequality_sign() == \
                        RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                    tail_type = TailType.LOWEST
                else:
                    tail_type = TailType.HIGHEST
            key = (attr, tail_type)
            if key in self.__sorted_scenario_prefix_sums:
                continue
            prefix_avgs = []
            for var_idx in range(self.__no_of_vars):
                arr = np.array(
                    self.__scenarios[attr][var_idx], dtype=np.float64)
                if tail_type == TailType.LOWEST:
                    sorted_arr = np.sort(arr)
                else:
                    sorted_arr = np.sort(arr)[::-1]
                k = np.arange(1, len(sorted_arr) + 1, dtype=np.float64)
                prefix_avgs.append(np.cumsum(sorted_arr) / k)
            self.__sorted_scenario_prefix_sums[key] = prefix_avgs
        if self.__optimize_lcvar and self.__query.get_objective().is_cvar_objective():
            obj = self.__query.get_objective()
            key = (obj.get_attribute_name(), obj.get_tail_type())
            if key not in self.__sorted_scenario_prefix_sums:
                attr = obj.get_attribute_name()
                tail_type = obj.get_tail_type()
                prefix_avgs = []
                for var_idx in range(self.__no_of_vars):
                    arr = np.array(
                        self.__scenarios[attr][var_idx], dtype=np.float64)
                    if tail_type == TailType.LOWEST:
                        sorted_arr = np.sort(arr)
                    else:
                        sorted_arr = np.sort(arr)[::-1]
                    k = np.arange(1, len(sorted_arr) + 1, dtype=np.float64)
                    prefix_avgs.append(np.cumsum(sorted_arr) / k)
                self.__sorted_scenario_prefix_sums[key] = prefix_avgs


    def __get_cvar_constraint_coefficients(
        self, cvar_constraint: CVaRConstraint,
        no_of_scenarios_to_consider: int,
        total_no_of_scenarios: int
    ):
        attr = cvar_constraint.get_attribute_name()
        key = (attr, cvar_constraint.get_tail_type())

        if key in self.__sorted_scenario_prefix_sums:
            k = no_of_scenarios_to_consider
            prefix_avgs = self.__sorted_scenario_prefix_sums[key]
            if k <= len(prefix_avgs[0]):
                return [prefix_avgs[var][k - 1]
                        for var in range(self.__no_of_vars)]

        tuple_wise_heaps = []
        is_max_heap = True
        if cvar_constraint.get_tail_type() == TailType.HIGHEST:
            is_max_heap = False

        for _ in range(self.__no_of_vars):
            tuple_wise_heaps.append(
                Heap(is_max_heap))

        for scenario_no in range(total_no_of_scenarios):

            scenario_index = scenario_no % \
                self.__feasible_no_of_scenarios_to_store

            if scenario_index == 0 and scenario_no != 0:
                self.__add_feasible_no_of_scenarios(attr)

            for var in range(self.__no_of_vars):
                scenario_value = self.__scenarios[attr][var][
                    scenario_index]
                tuple_wise_heaps[var].push(scenario_value)
                if tuple_wise_heaps[var].size() > \
                    no_of_scenarios_to_consider:
                    tuple_wise_heaps[var].pop()

        return [tuple_wise_heaps[var].sum()/no_of_scenarios_to_consider\
                for var in range(self.__no_of_vars)]

    
    def __add_lcvar_constraint_to_model(
        self,
        risk_constraint: CVaRConstraint | VaRConstraint,
        cvarified_constraint: CVaRConstraint,
        no_of_scenarios: int,
        no_of_scenarios_to_consider: int
    ):
        coefficients = \
            self.__get_cvar_constraint_coefficients(
                cvarified_constraint,
                no_of_scenarios_to_consider,
                no_of_scenarios
            )
        
        gurobi_inequality = self.__get_gurobi_inequality(
            cvarified_constraint.get_inequality_sign()
        )

        self.__risk_constraints.append(risk_constraint)
        self.__risk_to_lcvar_constraint_mapping[
            risk_constraint] = \
                self.__model.addConstr(
                    gp.LinExpr(
                        coefficients, self.__vars),
                    gurobi_inequality,
                    cvarified_constraint.get_sum_limit()
                )


    def __get_cvarified_constraint(
        self, constraint: VaRConstraint,
        cvar_threshold: float
    ):
        cvarified_constraint = \
            CVaRConstraint()
        cvarified_constraint.set_percentage_of_scenarios(
            (1 - constraint.get_probability_threshold())*100)
        if constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            cvarified_constraint.set_tail_type('l')
        else:
            cvarified_constraint.set_tail_type('h')
                            
        if constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            cvarified_constraint.set_inequality_sign('>')
        else:
            cvarified_constraint.set_inequality_sign('<')
                            
        cvarified_constraint.set_attribute_name(
            constraint.get_attribute_name()
        )

        cvarified_constraint.set_sum_limit(
            cvar_threshold
        )

        return cvarified_constraint

    def __build_aggregate_z_lin_expr(self, group: frozenset, attribute: str):
        """Build LHS and RHS for aggregate Z_s constraint over a group of scenarios."""
        group_list = sorted(group)
        G = len(group_list)
        lhs = gp.LinExpr([1.0] * G, [self.__Zs[s] for s in group_list])
        agg_coeffs = [
            sum(self.__scenarios[attribute][i][s] for s in group_list)
            for i in range(self.__no_of_vars)
        ]
        rhs = gp.LinExpr(agg_coeffs, self.__vars)
        rhs.addTerms(-float(G), self.__V)
        return lhs, rhs

    def __init_z_lazy_constraints(self, no_of_scenarios: int, attribute: str):
        """Initialize lazy Z_s constraints with single aggregate over all scenarios."""
        self.__z_groups = []
        self.__z_group_constrs = []
        all_s = frozenset(range(no_of_scenarios))
        self.__z_groups.append(all_s)
        lhs, rhs = self.__build_aggregate_z_lin_expr(all_s, attribute)
        tail = self.__query.get_objective().get_tail_type()
        constr = self.__model.addConstr(lhs <= rhs if tail == TailType.LOWEST else lhs >= rhs)
        self.__z_group_constrs.append(constr)
        self.__z_lazy_active = True

    def __check_z_violations(self, attribute: str, tol: float = 1e-6) -> list[int]:
        """Vectorized check for violated individual Z_s constraints."""
        V_val = self.__V.x
        x_arr = np.array([v.x for v in self.__vars])
        mat = np.array([self.__scenarios[attribute][i]
                        for i in range(self.__no_of_vars)])
        lin_vals = mat.T @ x_arr
        rhs_vals = lin_vals - V_val
        z_vals = np.array([z.x for z in self.__Zs])
        tail = self.__query.get_objective().get_tail_type()
        if tail == TailType.LOWEST:
            mask = z_vals > rhs_vals + tol
        else:
            mask = z_vals < rhs_vals - tol
        return list(np.where(mask)[0])

    def __refine_z_constraints(self, attribute: str, tol: float = 1e-6) -> bool:
        """Perform one round of constraint splitting. Returns True if splits occurred."""
        violated_set = set(self.__check_z_violations(attribute, tol))
        if not violated_set:
            return False
        tail = self.__query.get_objective().get_tail_type()
        new_groups, new_constrs = [], []
        for group, constr in zip(self.__z_groups, self.__z_group_constrs):
            viol = group & violated_set
            sat  = group - violated_set
            print(f'Group of size {len(group)}: {len(viol)} violated, {len(sat)} satisfied')
            if not viol:
                new_groups.append(group)
                new_constrs.append(constr)
                continue
            self.__model.remove(constr)
            for sub in [sat, viol]:
                if not sub:
                    continue
                lhs, rhs = self.__build_aggregate_z_lin_expr(sub, attribute)
                c = self.__model.addConstr(lhs <= rhs if tail == TailType.LOWEST else lhs >= rhs)
                new_groups.append(sub)
                new_constrs.append(c)
        self.__z_groups = new_groups
        print(f'Added {len(new_constrs) - len(self.__z_group_constrs)} Z_s constraints; now have {len(self.__z_group_constrs)} total.')
        self.__z_group_constrs = new_constrs
        return True

    def __add_constraints_to_model(
        self, no_of_scenarios: int,
        no_of_scenarios_to_consider: list[int],
        probabilistically_constrained: bool,
        cvar_thresholds: list[float],
        trivial_constraints: list[int]
    ):
        self.__add_all_scenarios_if_possible(
            no_of_scenarios
        )
        any_high_tail_cvar = any(
            c.get_percentage_of_scenarios() > 50.0
            for c in self.__query.get_constraints()
            if c.is_cvar_constraint()
        )
        stochastic_constraint_index = 0
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
                threshold_override = None
                if probabilistically_constrained and \
                        stochastic_constraint_index not in trivial_constraints:
                    threshold_override = cvar_thresholds[
                        stochastic_constraint_index
                    ]
                expected_sum_model_constraint = \
                    self.__add_expected_sum_constraint_to_model(
                        constraint,
                        no_of_scenarios,
                        threshold_override=threshold_override
                    )
                if probabilistically_constrained and \
                        stochastic_constraint_index not in trivial_constraints:
                    self.__risk_to_lcvar_constraint_mapping[
                        constraint
                    ] = expected_sum_model_constraint
                stochastic_constraint_index += 1
                continue

            if constraint.is_risk_constraint():
                if probabilistically_constrained:
                    if stochastic_constraint_index not in \
                        trivial_constraints:
                        if constraint.is_cvar_constraint():
                            cvarified_constraint = \
                                constraint
                        else:
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint=constraint,
                                    cvar_threshold=cvar_thresholds[
                                        stochastic_constraint_index]
                                )
                        
                        self.__add_lcvar_constraint_to_model(
                            risk_constraint=constraint,
                            cvarified_constraint=cvarified_constraint,
                            no_of_scenarios=no_of_scenarios,
                            no_of_scenarios_to_consider=\
                                no_of_scenarios_to_consider[
                                    stochastic_constraint_index
                                ]
                        )
                else:
                    if not any_high_tail_cvar:
                        attribute = constraint.get_attribute_name()
                        sum_limit = constraint.get_sum_limit()
                        self.__add_feasible_no_of_scenarios(attribute)
                        coefficients = []
                        for tuple_scenarios in self.__scenarios[attribute]:
                            coefficients.append(np.median(tuple_scenarios))

                        gurobi_inequality = GRB.LESS_EQUAL
                        if constraint.get_inequality_sign() == \
                                RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                            gurobi_inequality = GRB.GREATER_EQUAL
                        self.__model.addConstr(
                            gp.LinExpr(coefficients, self.__vars),
                            gurobi_inequality, sum_limit
                        )

                stochastic_constraint_index += 1

        if not self.__check_feasibility and not self.__optimize_lcvar and \
                self.__query.get_objective().is_cvar_objective():
            attribute = self.__query.get_objective().get_attribute_name()
            self.__z_lazy_active = False
            self.__init_z_lazy_constraints(no_of_scenarios, attribute)


    
    def __add_objective_to_model(
        self, objective: Objective,
        no_of_scenarios: int):
        if self.__check_feasibility:
            if objective.is_cvar_objective():
                attr = objective.get_attribute_name()
                coefficients = [
                    np.average(self.__scenarios[attr][idx])
                    for idx in range(self.__no_of_vars)
                ]
                gurobi_dir = (
                    GRB.MAXIMIZE
                    if objective.get_objective_type() == ObjectiveType.MAXIMIZATION
                    else GRB.MINIMIZE
                )
                self.__model.setObjective(
                    gp.LinExpr(coefficients, self.__vars), gurobi_dir)
            else:
                raise NotImplementedError(
                    'check_feasibility=True is only supported for CVaR objectives.'
                )
            return

        if objective.is_cvar_objective():
            if self.__optimize_lcvar:
                synthetic = CVaRConstraint()
                synthetic.set_attribute_name(objective.get_attribute_name())
                synthetic.set_tail_type(
                    'l' if objective.get_tail_type() == TailType.LOWEST else 'h')
                synthetic.set_percentage_of_scenarios(
                    objective.get_percentage_of_scenarios() * 100)
                n_tail = max(1, int(np.floor(
                    objective.get_percentage_of_scenarios() * no_of_scenarios)))
                coefficients = self.__get_cvar_constraint_coefficients(
                    synthetic, n_tail, no_of_scenarios)
                gurobi_dir = (GRB.MAXIMIZE
                              if objective.get_objective_type() == ObjectiveType.MAXIMIZATION
                              else GRB.MINIMIZE)
                self.__model.setObjective(
                    gp.LinExpr(coefficients, self.__vars), gurobi_dir)
                return

            is_maximize = (objective.get_objective_type() == ObjectiveType.MAXIMIZATION)
            tail_is_lowest = (objective.get_tail_type() == TailType.LOWEST)
            if is_maximize != tail_is_lowest:
                raise NotImplementedError(
                    'Direct CVaR LP optimization is only supported for '
                    'MAXIMIZE+LOWEST and MINIMIZE+HIGHEST tail. '
                    'MINIMIZE+LOWEST and MAXIMIZE+HIGHEST are non-convex '
                    '(minimizing a concave / maximizing a convex function) '
                    'and cannot be expressed as a single LP. '
                    'Use CVaROptimizerBaseline (check_feasibility=True) instead.'
                )
            coefficients = [1]
            for _ in range(len(self.__Zs)):
                coefficients.append(1.0/(
                    objective.get_percentage_of_scenarios() *\
                        no_of_scenarios))

            if objective.get_objective_type() == ObjectiveType.MINIMIZATION:
                self.__model.setObjective(
                    gp.LinExpr(coefficients, [self.__V] + self.__Zs),
                    GRB.MINIMIZE
                )
                return

            self.__model.setObjective(
                gp.LinExpr(coefficients, [self.__V] + self.__Zs),
                GRB.MAXIMIZE
            )
            return      
        
        attr = objective.get_attribute_name()
        coefficients = []

        if objective.get_stochasticity() == \
            Stochasticity.DETERMINISTIC:
            coefficients = self.__values[attr]
        else:
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
    

    def __model_setup(
        self, no_of_scenarios: int,
        no_of_scenarios_to_consider: list[int],
        probabilistically_constrained = False,
        cvar_lower_bounds = [],
        trivial_constraints = []
    ):
        
        self.__model = gp.Model(
            env=self.__gurobi_env)
        if self.__check_feasibility:
            self.__model.Params.SolutionLimit = 1
            self.__model.Params.MIPFocus = 1
        self.__add_variables_to_model(no_of_scenarios)
        self.__add_constraints_to_model(
            no_of_scenarios,
            no_of_scenarios_to_consider,
            probabilistically_constrained,
            cvar_lower_bounds, trivial_constraints)
        self.__add_objective_to_model(
            self.__query.get_objective(),
            no_of_scenarios)
        

    def __extract_package_from_model(self):
        """Extract solution from model variables."""
        package_dict = {}
        idx = 0
        try:
            for var in self.__vars:
                if var.x > 0:
                    package_dict[self.__ids[idx]] = var.x
                idx += 1
        except AttributeError:
            return None
        if not package_dict:
            return None
        return package_dict

    def __get_package_with_z_refinement(self):
        """Solve with lazy Z_s constraint refinement."""
        attribute = self.__query.get_objective().get_attribute_name()
        max_rounds = len(self.__Zs) + 1
        for _ in range(max_rounds):
            self.__metrics.start_optimizer()
            self.__model.optimize()
            self.__metrics.end_optimizer()
            status = self.__model.Status
            if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
                return None
            if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                return self.__extract_package_from_model()
            try:
                _ = self.__V.x
            except AttributeError:
                return None
            if not self.__refine_z_constraints(attribute):
                break
            print('Needs more refinement, adding more Z_s constraints and re-optimizing...')
        return self.__extract_package_from_model()

    def __get_package(self):
        if (not self.__check_feasibility and not self.__optimize_lcvar
                and self.__query.get_objective().is_cvar_objective()
                and self.__z_lazy_active):
            return self.__get_package_with_z_refinement()
        self.__metrics.start_optimizer()
        self.__model.optimize()
        self.__metrics.end_optimizer()
        return self.__extract_package_from_model()
    

    def __get_package_with_indices(self):
        package_dict = {}
        idx = 0
        try:
            for var in self.__vars:
                if var.x > 0:
                    package_dict[
                        idx] = var.x
                idx += 1
        except AttributeError:
            return None
        return package_dict
    

    def __get_objective_value_among_optimization_scenarios(
        self, package_with_indices, no_of_scenarios
    ) -> float:
        if package_with_indices is None:
            return 0
        attr = self.__query.get_objective().get_attribute_name()
        if no_of_scenarios <= \
            self.__feasible_no_of_scenarios_to_store:
            if self.__query.get_objective().is_cvar_objective():
                idxs = list(package_with_indices.keys())
                mults = np.array([package_with_indices[i] for i in idxs])
                mat = np.array([self.__scenarios[attr][i] for i in idxs])
                scenario_scores = mat.T @ mults
                tail_type = self.__query.get_objective().get_tail_type()
                if tail_type == TailType.HIGHEST:
                    scenario_scores = np.sort(scenario_scores)[::-1]
                else:
                    scenario_scores = np.sort(scenario_scores)
                k = max(1, int(np.floor(
                    self.__query.get_objective().get_percentage_of_scenarios() *
                    no_of_scenarios
                )))
                return float(np.average(scenario_scores[:k]))
            sum = 0
            for idx in package_with_indices:
                sum += np.average(self.__scenarios[attr][idx]) * \
                    package_with_indices[idx]
            return sum
        
        if self.__query.get_objective().is_cvar_objective():
            all_scenario_scores = []
            total_scenarios = 0
            while total_scenarios < no_of_scenarios:
                self.__add_feasible_no_of_scenarios(attr)
                total_scenarios += self.__feasible_no_of_scenarios_to_store
                idxs = list(package_with_indices.keys())
                mults = np.array([package_with_indices[i] for i in idxs])
                mat = np.array([self.__scenarios[attr][i] for i in idxs])
                all_scenario_scores.extend((mat.T @ mults).tolist())
            scenario_scores = np.array(all_scenario_scores)
            tail_type = self.__query.get_objective().get_tail_type()
            if tail_type == TailType.HIGHEST:
                scenario_scores = np.sort(scenario_scores)[::-1]
            else:
                scenario_scores = np.sort(scenario_scores)
            k = max(1, int(np.floor(
                self.__query.get_objective().get_percentage_of_scenarios() * total_scenarios
            )))
            return float(np.average(scenario_scores[:k]))

        sum = 0
        total_scenarios = 0
        while total_scenarios < no_of_scenarios:
            self.__add_feasible_no_of_scenarios(attr)
            total_scenarios += self.__feasible_no_of_scenarios_to_store
            for idx in package_with_indices:
                for value in self.__scenarios[attr][idx]:
                    sum += value * package_with_indices[idx]
        return sum / total_scenarios
    

    def __is_objective_value_relative_diff_high(
        self, package, package_with_indices,
        no_of_scenarios
    ) -> bool:
        if self.__check_feasibility:
            return False, 0.0
        if package is None:
            return False, 0
        objective_value_optimization_scenarios =\
            self.__get_objective_value_among_optimization_scenarios(
                package_with_indices, no_of_scenarios
            )
        #print('Obj value optimization:',
        #      objective_value_optimization_scenarios)
        objective_value_validation_scenarios =\
            self.__validator.get_validation_objective_value(
                package
            )
        #print('Obj value validation:',
        #      objective_value_validation_scenarios)
        diff = objective_value_optimization_scenarios - \
                    objective_value_validation_scenarios
        
        if diff < 0:
            diff *= -1

        rel_diff = diff / (objective_value_validation_scenarios
            + 0.00001)
        
        #print('Relative difference:', rel_diff)
        if rel_diff > self.__sampling_tolerance:
            return True, objective_value_validation_scenarios
        return False, objective_value_validation_scenarios
    

    def __is_objective_value_enough(
        self, objective_value: float,
        objective_upper_bound: float
    ):
        if self.__check_feasibility:
            return True
        objective = self.__query.get_objective()

        if objective.get_objective_type() == \
            ObjectiveType.MAXIMIZATION:
            if objective_upper_bound >= 0:
                return objective_value >= \
                    (1 - self.__approximation_bound) *\
                        objective_upper_bound
            else:
                return objective_value >= \
                    (1 + self.__approximation_bound) *\
                        objective_upper_bound
        
        if objective_upper_bound >= 0:
            return objective_value <= \
                (1 + self.__approximation_bound) *\
                    objective_upper_bound

        return objective_value <= \
                (1 - self.__approximation_bound) *\
                    objective_upper_bound
    

    def __get_scenario_scores_ascending(
        self, package_with_indices,
        no_of_scenarios: int,
        constraint: VaRConstraint | CVaRConstraint
    ) -> list[float]:
        attr = constraint.get_attribute_name()
        cache_key = (attr, tuple(sorted(package_with_indices.items())), no_of_scenarios)
        if cache_key in self.__opt_scenario_scores_cache:
            return self.__opt_scenario_scores_cache[cache_key]

        if no_of_scenarios <= \
            self.__feasible_no_of_scenarios_to_store:
            idxs = list(package_with_indices.keys())
            mults = np.array([package_with_indices[i] for i in idxs])
            mat = np.array([self.__scenarios[attr][i] for i in idxs])
            scenario_scores = np.sort(mat.T @ mults).tolist()
            self.__opt_scenario_scores_cache[cache_key] = scenario_scores
            return scenario_scores

        scenario_scores = []
        total_scenarios = 0
        while total_scenarios < no_of_scenarios:
            self.__add_feasible_no_of_scenarios(attr)
            total_scenarios += \
                self.__feasible_no_of_scenarios_to_store
            idxs = list(package_with_indices.keys())
            mults = np.array([package_with_indices[i] for i in idxs])
            mat = np.array([self.__scenarios[attr][i] for i in idxs])
            scenario_scores.extend((mat.T @ mults).tolist())

        scenario_scores.sort()
        return scenario_scores
        
    
    def __get_cvar_among_optimization_scenarios(
        self, package_with_indices,
        no_of_scenarios: int,
        constraint: CVaRConstraint
    ) -> float:
        scenario_scores = \
            self.__get_scenario_scores_ascending(
                package_with_indices, no_of_scenarios,
                constraint
            )
        if constraint.get_tail_type() == TailType.HIGHEST:
            scenario_scores.reverse()
        scenarios_to_consider = \
            max(1, int(np.floor((constraint.get_percentage_of_scenarios()\
                      *no_of_scenarios)/100)))
        return np.average(
            scenario_scores[0:scenarios_to_consider])
    

    def __get_var_among_optimization_scenarios(
        self, package_with_indices,
        no_of_scenarios: int,
        constraint: VaRConstraint
    ) -> float:
        scenario_scores = \
            self.__get_scenario_scores_ascending(
                package_with_indices, no_of_scenarios,
                constraint
            )
        scenario_scores.reverse()
        scenarios_to_consider = \
            min(
                int(np.floor((constraint.get_probability_threshold()*\
                          no_of_scenarios))),
                no_of_scenarios - 1
            )
        return scenario_scores[scenarios_to_consider]
    

    def __is_var_relative_difference_high(
        self, package_with_indices,
        package, no_of_scenarios,
        constraint
    ) -> bool:
        if self.__check_feasibility:
            return False, constraint.get_sum_limit()
        if package is None:
            return False, constraint.get_sum_limit()
        
        var_optimization = \
            self.__get_var_among_optimization_scenarios(
                package_with_indices, no_of_scenarios,
                constraint
            )

        var_validation = \
            self.__validator.get_var_among_validation_scenarios(
                package, constraint
            )
        

        if constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            rel_diff = var_optimization - var_validation
        else:
            rel_diff= var_validation - var_optimization
        rel_diff /= (var_validation + 0.000001)

        if rel_diff > self.__sampling_tolerance:
            return True, var_validation
        return False, var_validation
    

    def __is_var_constraint_satisfied(
        self, var_constraint: VaRConstraint,
        var_validation: float
    ) -> bool:
        if var_constraint.get_inequality_sign() ==\
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            return var_validation >=\
                var_constraint.get_sum_limit()
        
        return var_validation <=\
            var_constraint.get_sum_limit()

    
    def __is_cvar_relative_difference_high(
        self, package_with_indices,
        package, no_of_scenarios,
        constraint
    ) -> bool:
        if self.__check_feasibility:
            return False, constraint.get_sum_limit()
        if package is None:
            return False, constraint.get_sum_limit()
        
        cvar_optimization = \
            self.__get_cvar_among_optimization_scenarios(
                package_with_indices, no_of_scenarios,
                constraint
            )
        cvar_validation = \
            self.__validator.get_cvar_constraint_satisfaction(
                package, constraint
            )
        
        if constraint.get_inequality_sign() == \
            RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
            rel_diff = cvar_optimization - cvar_validation
        else:
            rel_diff= cvar_validation - cvar_optimization
        
        rel_diff /= (cvar_validation + 0.0000001)
        
        if rel_diff > self.__sampling_tolerance:
            return True, cvar_validation
        return False, cvar_validation
    

    def __is_cvar_constraint_satisfied(
        self, cvar_constraint: CVaRConstraint,
        cvar_validation: float
    ) -> bool:
        if cvar_constraint.get_inequality_sign() == \
            RelationalOperators.LESS_THAN_OR_EQUAL_TO:
            return (cvar_validation <=\
                        cvar_constraint.get_sum_limit())
        
        return (cvar_validation >=\
                    cvar_constraint.get_sum_limit())

    
    def __l_inf(self, l1: list[int | float],
                l2: list[int | float]) -> float:
        if len(l1) != len(l2):
            raise Exception
        max_abs_diff = 0

        for ind in range(len(l1)):
            diff = l1[ind] - l2[ind]
            if diff < 0:
                diff *= -1
            if diff > max_abs_diff:
                max_abs_diff = diff
        
        return max_abs_diff


    def __repair_lower_bound_anchors(
        self,
        no_of_scenarios: int,
        cvar_upper_bounds: list[float | int],
        cvar_lower_bounds: list[float | int],
        trivial_constraints: list[int],
        no_of_scenarios_to_consider: list[int],
        can_add_scenarios: bool
    ) -> tuple[list[float | int], bool, bool, object, float | None]:
        cvar_lower_bounds = list(cvar_lower_bounds)
        if len(cvar_lower_bounds) == 0:
            return cvar_lower_bounds, False, False, None, None

        print('[LowerRepair] Starting | upper bounds:', cvar_upper_bounds,
              '| lower bounds:', cvar_lower_bounds,
              '| scenarios:', no_of_scenarios_to_consider)
        is_model_setup = False
        best_feasible_package = None
        best_feasible_objective = None
        repair_iterations = 0

        while True:
            print('[LowerRepair] Trying lower bounds:', cvar_lower_bounds)
            if not is_model_setup:
                self.__model_setup(
                    no_of_scenarios=no_of_scenarios,
                    no_of_scenarios_to_consider=no_of_scenarios_to_consider,
                    probabilistically_constrained=True,
                    cvar_lower_bounds=cvar_lower_bounds,
                    trivial_constraints=trivial_constraints
                )
                is_model_setup = True
            else:
                stochastic_constraint_index = 0
                for constraint in self.__get_searchable_constraints():
                    if stochastic_constraint_index not in trivial_constraints:
                        self.__risk_to_lcvar_constraint_mapping[
                            constraint].rhs = cvar_lower_bounds[
                                stochastic_constraint_index]
                    stochastic_constraint_index += 1

            package = self.__get_package()
            print('[LowerRepair] Package:', package)
            if package is None:
                print('[LowerRepair] No package found; keeping current'
                      ' lower bounds as heuristic anchors.')
                return cvar_lower_bounds, is_model_setup, False, best_feasible_package, best_feasible_objective

            package_with_indices = self.__get_package_with_indices()

            unacceptable_diff, _ = \
                self.__is_objective_value_relative_diff_high(
                    package, package_with_indices,
                    no_of_scenarios
                )
            if unacceptable_diff and can_add_scenarios:
                print('[LowerRepair] Needs more scenarios (obj diff)')
                return cvar_lower_bounds, is_model_setup, True, best_feasible_package, best_feasible_objective

            violated_constraints = []
            stochastic_constraint_index = 0
            for constraint in self.__get_searchable_constraints():
                is_trivial = stochastic_constraint_index in trivial_constraints
                if is_trivial:
                    stochastic_constraint_index += 1
                    continue

                if constraint.is_expected_sum_constraint():
                    satisfied = \
                        self.__validator.get_expected_sum_constraint_feasibility(
                            package, constraint
                        )
                elif constraint.is_cvar_constraint():
                    unacceptable_diff, cvar_validation = \
                        self.__is_cvar_relative_difference_high(
                            package_with_indices, package,
                            no_of_scenarios, constraint
                        )
                    if unacceptable_diff and can_add_scenarios:
                        print('[LowerRepair] Needs more scenarios'
                              ' (CVaR diff)')
                        return cvar_lower_bounds, is_model_setup, True, best_feasible_package, best_feasible_objective
                    satisfied = self.__is_cvar_constraint_satisfied(
                        constraint, cvar_validation
                    )
                else:
                    unacceptable_diff, var_validation = \
                        self.__is_var_relative_difference_high(
                            package_with_indices, package,
                            no_of_scenarios, constraint
                        )
                    if unacceptable_diff and can_add_scenarios:
                        print('[LowerRepair] Needs more scenarios'
                              ' (VaR diff)')
                        return cvar_lower_bounds, is_model_setup, True, best_feasible_package, best_feasible_objective
                    satisfied = self.__is_var_constraint_satisfied(
                        constraint, var_validation
                    )

                if not satisfied:
                    violated_constraints.append(stochastic_constraint_index)
                stochastic_constraint_index += 1

            if len(violated_constraints) == 0:
                print('[LowerRepair] Lower bounds validated.')
                if best_feasible_package is None:
                    best_feasible_package = package
                    best_feasible_objective = self.__validator.get_validation_objective_value(package)
                else:
                    cand_obj = self.__validator.get_validation_objective_value(package)
                    if (self.__query.get_objective().get_objective_type() == ObjectiveType.MAXIMIZATION and cand_obj > best_feasible_objective) or \
                       (self.__query.get_objective().get_objective_type() != ObjectiveType.MAXIMIZATION and cand_obj < best_feasible_objective):
                        best_feasible_package = package
                        best_feasible_objective = cand_obj
                return cvar_lower_bounds, is_model_setup, False, best_feasible_package, best_feasible_objective

            print('[LowerRepair] Violated constraint indices:',
                  violated_constraints)
            for violated_index in violated_constraints:
                diff = abs(
                    cvar_upper_bounds[violated_index] -
                    cvar_lower_bounds[violated_index]
                )
                if cvar_lower_bounds[violated_index] <= \
                        cvar_upper_bounds[violated_index]:
                    cvar_lower_bounds[violated_index] -= diff
                else:
                    cvar_lower_bounds[violated_index] += diff
            print('[LowerRepair] Updated lower bounds:',
                  cvar_lower_bounds)
            repair_iterations += 1
            if repair_iterations >= 10:
                print('[LowerRepair] Max iterations reached; requesting more scenarios.')
                return cvar_lower_bounds, is_model_setup, True, best_feasible_package, best_feasible_objective

    def __cvar_threshold_search(
        self, no_of_scenarios: int,
        cvar_upper_bounds: list[float | int],
        cvar_lower_bounds: list[float | int],
        trivial_constraints: list[int],
        objective_upper_bound: float,
        no_of_scenarios_to_consider: list[int],
        is_model_setup: bool,
        can_add_scenarios: bool
    ) -> CVaRificationSearchResults:
        cvar_upper_bounds = list(cvar_upper_bounds)
        cvar_lower_bounds = list(cvar_lower_bounds)
        best_feasible_package = None
        best_feasible_objective = None
        found_non_none_infeasible = False
        if not is_model_setup and \
                self.__l_inf(cvar_upper_bounds, cvar_lower_bounds) < \
                self.__bisection_threshold:
            cvar_initial_thresholds = [
                (u + l) / 2.0
                for u, l in zip(cvar_upper_bounds, cvar_lower_bounds)
            ]
            self.__model_setup(
                no_of_scenarios=no_of_scenarios,
                no_of_scenarios_to_consider=no_of_scenarios_to_consider,
                probabilistically_constrained=True,
                cvar_lower_bounds=cvar_initial_thresholds,
                trivial_constraints=trivial_constraints
            )
            is_model_setup = True
        while self.__l_inf(
            cvar_upper_bounds, cvar_lower_bounds) >= \
            self.__bisection_threshold:

            cvar_mid_thresholds = []
            for ind in range(len(cvar_lower_bounds)):
                cvar_mid_thresholds.append(
                    (cvar_lower_bounds[ind] + \
                    cvar_upper_bounds[ind]) / 2.0
                )
            
            print('Bisecting for thresholds')
            print('CVaR upper bounds:', cvar_upper_bounds)
            print('CVaR lower bounds:', cvar_lower_bounds)
            print('CVaR mid thresholds:', cvar_mid_thresholds)

            if not is_model_setup:
                self.__model_setup(
                    no_of_scenarios=no_of_scenarios,
                    no_of_scenarios_to_consider=\
                        no_of_scenarios_to_consider,
                    probabilistically_constrained=True,
                    cvar_lower_bounds=cvar_mid_thresholds,
                    trivial_constraints=trivial_constraints
                )
                is_model_setup = True
            
            else:
                stochastic_constraint_index = 0
                for constraint in self.__get_searchable_constraints():
                    if stochastic_constraint_index not in \
                        trivial_constraints:
                        self.__risk_to_lcvar_constraint_mapping[
                            constraint].rhs = cvar_mid_thresholds[
                                stochastic_constraint_index]
                    stochastic_constraint_index += 1
            
            package = self.__get_package()
            print('Package:', package)
            if package is None:
                cvar_lower_bounds = cvar_mid_thresholds
                continue
            package_with_indices = \
                self.__get_package_with_indices()
            print('Packages with indices:', package_with_indices)
            
            unacceptable_diff, validation_objective_value = \
                self.__is_objective_value_relative_diff_high(
                    package, package_with_indices,
                    no_of_scenarios
                )
            
            if unacceptable_diff and can_add_scenarios:
                print('Unacceptable diff for objective')
                result = CVaRificationSearchResults()
                result.set_needs_more_scenarios(True)
                return result
            
            all_constraints_satisfied = True
            all_constraints_violated = True
            stochastic_constraint_index = 0

            for constraint in self.__get_searchable_constraints():
                is_trivial = stochastic_constraint_index in trivial_constraints
                if constraint.is_expected_sum_constraint():
                    expected_sum_satisfied = \
                        self.__validator.get_expected_sum_constraint_feasibility(
                            package, constraint
                        )
                    if expected_sum_satisfied:
                        print('Expected sum constraint satisfied')
                        all_constraints_violated = False
                        if not is_trivial:
                            cvar_lower_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                    else:
                        print('Expected sum constraint not satisfied')
                        all_constraints_satisfied = False
                        if not is_trivial:
                            cvar_upper_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                    stochastic_constraint_index += 1
                    continue

                if not is_trivial:
                    if constraint.is_cvar_constraint():
                        unacceptable_diff, cvar_validation = \
                            self.__is_cvar_relative_difference_high(
                                package_with_indices, package,
                                no_of_scenarios, constraint
                            )
                        if unacceptable_diff and can_add_scenarios:
                            result = CVaRificationSearchResults()
                            result.set_needs_more_scenarios(True)
                            return result
                        if self.__is_cvar_constraint_satisfied(
                            constraint, cvar_validation
                        ):
                            print('CVaR constraint satisfied')
                            all_constraints_violated = False
                            cvar_lower_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                        else:
                            print('CVaR constraint not satisfied')
                            all_constraints_satisfied = False
                            cvar_upper_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]

                    if constraint.is_var_constraint():
                        unacceptable_diff, var_validation = \
                            self.__is_var_relative_difference_high(
                                package_with_indices, package,
                                no_of_scenarios, constraint
                            )
                        if unacceptable_diff and can_add_scenarios:
                            result = CVaRificationSearchResults()
                            result.set_needs_more_scenarios(True)
                            return result
                        if self.__is_var_constraint_satisfied(
                            constraint, var_validation
                        ):
                            print('VaR constraint satisfied')
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint,
                                    constraint.get_sum_limit()
                                )
                            cvarified_cvar_validation = \
                                self.__validator.get_cvar_constraint_satisfaction(
                                    package, cvarified_constraint
                                )
                            if self.__is_cvar_constraint_satisfied(
                                cvarified_constraint,
                                cvarified_cvar_validation
                            ):
                                all_constraints_violated = False
                            cvar_lower_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                        else:
                            print('VaR constraint not satisfied')
                            all_constraints_satisfied = False
                            cvar_upper_bounds[stochastic_constraint_index] = \
                                cvar_mid_thresholds[stochastic_constraint_index]
                stochastic_constraint_index += 1

            if all_constraints_satisfied:
                if self.__is_objective_value_enough(
                    validation_objective_value, objective_upper_bound
                ):
                    result = CVaRificationSearchResults()
                    result.set_found_appropriate_package(True)
                    result.set_package(package)
                    result.set_objective_value(
                        validation_objective_value)
                    return result
                else:
                    if best_feasible_package is None or \
                        (self.__query.get_objective().get_objective_type() == \
                            ObjectiveType.MAXIMIZATION and \
                            validation_objective_value > best_feasible_objective) or \
                        (self.__query.get_objective().get_objective_type() != \
                            ObjectiveType.MAXIMIZATION and \
                            validation_objective_value < best_feasible_objective):
                        best_feasible_package = package
                        best_feasible_objective = validation_objective_value
            else:
                # Package is not None but constraints are not satisfied
                if package is not None:
                    found_non_none_infeasible = True

            if all_constraints_violated:
                if self.__query.get_objective().get_objective_type() ==\
                    ObjectiveType.MAXIMIZATION:
                    if validation_objective_value < objective_upper_bound:
                        objective_upper_bound = validation_objective_value
                else:
                    if validation_objective_value > objective_upper_bound:
                        objective_upper_bound = validation_objective_value

        if best_feasible_package is not None and \
            self.__is_objective_value_enough(
                best_feasible_objective, objective_upper_bound):
            result = CVaRificationSearchResults()
            result.set_found_appropriate_package(True)
            result.set_package(best_feasible_package)
            result.set_objective_value(best_feasible_objective)
            return result

        result = CVaRificationSearchResults()
        result.set_objective_upper_bound(
            objective_upper_bound)
        result.set_cvar_thresholds(
            cvar_lower_bounds)
        if best_feasible_package is not None:
            result.set_best_feasible_package(best_feasible_package)
            result.set_best_feasible_objective(best_feasible_objective)
        result.set_found_non_none_infeasible(found_non_none_infeasible)
        return result

    def __cvar_coefficient_search(
        self, no_of_scenarios: int,
        cvar_thresholds: list[float | int],
        trivial_constraints: list[int],
        objective_upper_bound: float,
        min_no_of_scenarios_to_consider: list[int],
        max_no_of_scenarios_to_consider: list[int],
        is_model_setup: bool,
        can_add_scenarios: bool
    ) -> CVaRificationSearchResults:
        max_no_of_scenarios_to_consider = list(max_no_of_scenarios_to_consider)
        best_feasible_package = None
        best_feasible_objective = None
        found_non_none_infeasible = False
        print('[CoeffSearch] Starting | scenarios:', no_of_scenarios,
              '| CVaR thresholds:', cvar_thresholds,
              '| obj upper bound:', objective_upper_bound,
              '| min scenarios:', min_no_of_scenarios_to_consider,
              '| max scenarios:', max_no_of_scenarios_to_consider)
        while self.__l_inf(
            min_no_of_scenarios_to_consider,
            max_no_of_scenarios_to_consider
        ) > 1:
            
            mid_no_of_scenarios_to_consider = []
            for ind in range(len(min_no_of_scenarios_to_consider)):
                mid_no_of_scenarios_to_consider.append(
                    (min_no_of_scenarios_to_consider[ind] +\
                     max_no_of_scenarios_to_consider[ind]) // 2
                )
            print('[CoeffSearch] Bisect | min:', min_no_of_scenarios_to_consider,
                  '| mid:', mid_no_of_scenarios_to_consider,
                  '| max:', max_no_of_scenarios_to_consider)

            if not is_model_setup:
                self.__model_setup(
                    no_of_scenarios=no_of_scenarios,
                    no_of_scenarios_to_consider=\
                        mid_no_of_scenarios_to_consider,
                    probabilistically_constrained=True,
                    cvar_lower_bounds=cvar_thresholds,
                    trivial_constraints=trivial_constraints
                )
                is_model_setup = True
            else:
                stochastic_constraint_index = 0
                for constraint in self.__get_searchable_constraints():
                    if constraint.is_risk_constraint() and \
                            stochastic_constraint_index not in trivial_constraints:
                        if constraint.is_cvar_constraint():
                            cvarified_constraint = constraint
                        else:
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint, cvar_thresholds[
                                        stochastic_constraint_index])
                        coefficients = \
                            self.__get_cvar_constraint_coefficients(
                                cvarified_constraint,
                                mid_no_of_scenarios_to_consider[
                                    stochastic_constraint_index],
                                no_of_scenarios)
                    
                        for _ in range(len(self.__vars)):
                            self.__model.chgCoeff(
                                self.__risk_to_lcvar_constraint_mapping[
                                    constraint],
                                self.__vars[_], coefficients[_]
                            )
                    stochastic_constraint_index += 1
            
            package = self.__get_package()
            if package is None:
                print('[CoeffSearch] No package found, tightening min')
                min_no_of_scenarios_to_consider = \
                    mid_no_of_scenarios_to_consider
                continue

            package_with_indices = \
                self.__get_package_with_indices()

            unacceptable_diff, validation_objective_value = \
                self.__is_objective_value_relative_diff_high(
                    package, package_with_indices,
                    no_of_scenarios
                )
            print('[CoeffSearch] Package found | validation obj:',
                  validation_objective_value,
                  '| obj diff unacceptable:', unacceptable_diff)

            if unacceptable_diff and can_add_scenarios:
                print('[CoeffSearch] Needs more scenarios (obj diff)')
                result = CVaRificationSearchResults()
                result.set_needs_more_scenarios(True)
                return result

            all_constraints_satisfied = True
            all_constraints_violated = True
            stochastic_constraint_index = 0

            for constraint in self.__get_searchable_constraints():
                is_trivial = stochastic_constraint_index in trivial_constraints
                if constraint.is_expected_sum_constraint():
                    expected_sum_satisfied = \
                        self.__validator.get_expected_sum_constraint_feasibility(
                            package, constraint
                        )
                    print('[CoeffSearch] Expected sum constraint',
                          stochastic_constraint_index,
                          '| limit:', constraint.get_sum_limit(),
                          '| satisfied:', expected_sum_satisfied)
                    if expected_sum_satisfied:
                        all_constraints_violated = False
                    else:
                        all_constraints_satisfied = False
                    stochastic_constraint_index += 1
                    continue

                if not is_trivial:
                    if constraint.is_cvar_constraint():
                        unacceptable_diff, cvar_validation = \
                            self.__is_cvar_relative_difference_high(
                                package_with_indices, package,
                                no_of_scenarios, constraint
                            )
                        if unacceptable_diff and can_add_scenarios:
                            print('[CoeffSearch] Needs more scenarios'
                                  ' (CVaR diff) constraint',
                                  stochastic_constraint_index)
                            result = CVaRificationSearchResults()
                            result.set_needs_more_scenarios(True)
                            return result

                        cvar_satisfied = self.__is_cvar_constraint_satisfied(
                            constraint, cvar_validation
                        )
                        print('[CoeffSearch] CVaR constraint',
                              stochastic_constraint_index,
                              '| validation CVaR:', cvar_validation,
                              '| limit:', constraint.get_sum_limit(),
                              '| satisfied:', cvar_satisfied)
                        if cvar_satisfied:
                            all_constraints_violated = False
                            min_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index]
                        else:
                            all_constraints_satisfied = False
                            max_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index] - 1

                    if constraint.is_var_constraint():
                        unacceptable_diff, var_validation = \
                            self.__is_var_relative_difference_high(
                                package_with_indices, package,
                                no_of_scenarios, constraint
                            )
                        if unacceptable_diff and can_add_scenarios:
                            print('[CoeffSearch] Needs more scenarios'
                                  ' (VaR diff) constraint',
                                  stochastic_constraint_index)
                            result = CVaRificationSearchResults()
                            result.set_needs_more_scenarios(True)
                            return result

                        var_satisfied = self.__is_var_constraint_satisfied(
                            constraint, var_validation
                        )
                        print('[CoeffSearch] VaR constraint',
                              stochastic_constraint_index,
                              '| validation VaR:', var_validation,
                              '| limit:', constraint.get_sum_limit(),
                              '| satisfied:', var_satisfied)
                        if var_satisfied:
                            cvarified_constraint = \
                                self.__get_cvarified_constraint(
                                    constraint,
                                    constraint.get_sum_limit()
                                )
                            cvarified_cvar_validation = \
                                self.__validator.get_cvar_constraint_satisfaction(
                                    package, cvarified_constraint
                                )
                            if self.__is_cvar_constraint_satisfied(
                                cvarified_constraint,
                                cvarified_cvar_validation
                            ):
                                all_constraints_violated = False

                            min_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index]
                        else:
                            all_constraints_satisfied = False
                            max_no_of_scenarios_to_consider[stochastic_constraint_index] = \
                                mid_no_of_scenarios_to_consider[stochastic_constraint_index] - 1

                stochastic_constraint_index += 1

            print('[CoeffSearch] All satisfied:', all_constraints_satisfied,
                  '| all violated:', all_constraints_violated)
            if all_constraints_satisfied:
                if self.__is_objective_value_enough(
                    validation_objective_value, objective_upper_bound
                ):
                    print('[CoeffSearch] Found appropriate package'
                          ' | obj:', validation_objective_value)
                    result = CVaRificationSearchResults()
                    result.set_found_appropriate_package(True)
                    result.set_package(package)
                    result.set_objective_value(
                        validation_objective_value)
                    return result
                else:
                    print('[CoeffSearch] Feasible but obj not enough'
                          ' | obj:', validation_objective_value,
                          '| upper bound:', objective_upper_bound)
                    if best_feasible_package is None or \
                        (self.__query.get_objective().get_objective_type() == \
                            ObjectiveType.MAXIMIZATION and \
                            validation_objective_value > best_feasible_objective) or \
                        (self.__query.get_objective().get_objective_type() != \
                            ObjectiveType.MAXIMIZATION and \
                            validation_objective_value < best_feasible_objective):
                        best_feasible_package = package
                        best_feasible_objective = validation_objective_value
            else:
                # Package is not None but constraints are not satisfied
                if package is not None:
                    found_non_none_infeasible = True

            if all_constraints_violated:
                if self.__query.get_objective().get_objective_type() ==\
                    ObjectiveType.MAXIMIZATION:
                    if validation_objective_value < objective_upper_bound:
                        print('[CoeffSearch] All violated, tightening'
                              ' obj upper bound:', objective_upper_bound,
                              '->', validation_objective_value)
                        objective_upper_bound = validation_objective_value
                else:
                    if validation_objective_value > objective_upper_bound:
                        print('[CoeffSearch] All violated, tightening'
                              ' obj upper bound:', objective_upper_bound,
                              '->', validation_objective_value)
                        objective_upper_bound = validation_objective_value

        if best_feasible_package is not None and \
            self.__is_objective_value_enough(
                best_feasible_objective, objective_upper_bound):
            print('[CoeffSearch] Returning best feasible (post-loop)'
                  ' | obj:', best_feasible_objective)
            result = CVaRificationSearchResults()
            result.set_found_appropriate_package(True)
            result.set_package(best_feasible_package)
            result.set_objective_value(best_feasible_objective)
            return result

        print('[CoeffSearch] Exiting without solution'
              ' | obj upper bound:', objective_upper_bound,
              '| min scenarios:', min_no_of_scenarios_to_consider)
        result = CVaRificationSearchResults()
        result.set_objective_upper_bound(objective_upper_bound)
        result.set_scenarios_to_consider(min_no_of_scenarios_to_consider)
        if best_feasible_package is not None:
            result.set_best_feasible_package(best_feasible_package)
            result.set_best_feasible_objective(best_feasible_objective)
        result.set_found_non_none_infeasible(found_non_none_infeasible)
        return result


    def __get_linearized_cvar_among_optimization_scenarios(
        self, probabilistically_unconstrained_package,
        cvar_constraint, no_of_scenarios
    ):
        no_of_scenarios_to_consider = \
            self.__get_no_of_scenarios_to_consider(
                cvar_constraint, no_of_scenarios
            )
        coefficients = self.__get_cvar_constraint_coefficients(
            cvar_constraint,
            no_of_scenarios_to_consider,
            no_of_scenarios
        )

        linearized_cvar = 0
        idx = 0
        for id in self.__ids:
            if id in probabilistically_unconstrained_package:
                linearized_cvar += \
                    coefficients[idx] *\
                        probabilistically_unconstrained_package[id]
            idx += 1
        return linearized_cvar
    
    def __get_expected_sum_among_optimization_scenarios(
        self, probabilistically_unconstrained_package,
        attribute
    ):
        expected_sum = 0
        for idx, id in enumerate(self.__ids):
            if id in probabilistically_unconstrained_package:
                expected_sum += np.mean(
                    self.__scenarios[attribute][idx])*\
                        probabilistically_unconstrained_package[id]
        return expected_sum

    def __get_bounds_for_risk_constraints(
        self, no_of_scenarios,
        probabilistically_unconstrained_package
    ):
        self.__add_all_scenarios_if_possible(
            no_of_scenarios)
        stochastic_constraint_index = 0
        cvar_lower_bounds = []
        cvar_upper_bounds = []
        min_no_of_scenarios_to_consider = []
        max_no_of_scenarios_to_consider = []
        trivial_constraints = []
        for constraint in self.__get_searchable_constraints():
            is_satisfied = False
            if constraint.is_expected_sum_constraint():
                if self.__validator.get_expected_sum_constraint_feasibility(
                    probabilistically_unconstrained_package,
                    constraint
                ):
                    is_satisfied = True
            elif constraint.is_var_constraint():
                if self.__validator.get_var_constraint_feasibility(
                    probabilistically_unconstrained_package,
                    constraint
                ):
                    is_satisfied = True
            elif constraint.is_cvar_constraint():
                if self.__validator.get_cvar_constraint_feasibility(
                    probabilistically_unconstrained_package,
                    constraint
                ):
                    is_satisfied = True

            if is_satisfied:
                trivial_constraints.append(stochastic_constraint_index)
                cvar_lower_bounds.append(constraint.get_sum_limit())
                cvar_upper_bounds.append(constraint.get_sum_limit())
                if constraint.is_expected_sum_constraint():
                    no_of_scenarios_to_consider = 1
                elif constraint.is_cvar_constraint():
                    no_of_scenarios_to_consider = \
                        max(1, int(np.floor(
                            constraint.get_percentage_of_scenarios() *
                            no_of_scenarios / 100.0
                        )))
                else:
                    no_of_scenarios_to_consider = \
                        max(1, int(np.ceil(
                            (1 - constraint.get_probability_threshold()) *
                            no_of_scenarios
                        )))
                min_no_of_scenarios_to_consider.append(
                    no_of_scenarios_to_consider
                )
                max_no_of_scenarios_to_consider.append(
                    no_of_scenarios_to_consider
                )
                stochastic_constraint_index += 1
                continue

            if constraint.is_expected_sum_constraint():
                expected_sum = \
                    self.__get_expected_sum_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        constraint.get_attribute_name()
                    )
                if constraint.get_inequality_sign() == \
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO:
                    cvar_lower_bounds.append(
                        min(expected_sum, constraint.get_sum_limit())
                    )
                    cvar_upper_bounds.append(constraint.get_sum_limit())
                else:
                    cvar_lower_bounds.append(constraint.get_sum_limit())
                    cvar_upper_bounds.append(
                        max(expected_sum, constraint.get_sum_limit())
                    )
                min_no_of_scenarios_to_consider.append(1)
                max_no_of_scenarios_to_consider.append(1)
                stochastic_constraint_index += 1
                continue

            if constraint.is_cvar_constraint():
                cvar_threshold = \
                    self.__get_linearized_cvar_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        constraint, no_of_scenarios
                    )
                if constraint.get_inequality_sign() == \
                        RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                    cvar_upper_bounds.append(
                        max(cvar_threshold, constraint.get_sum_limit()))
                else:
                    cvar_upper_bounds.append(cvar_threshold)
                expected_sum = \
                    self.__get_expected_sum_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        constraint.get_attribute_name()
                    )
                if constraint.get_inequality_sign() == \
                        RelationalOperators.LESS_THAN_OR_EQUAL_TO:
                    cvar_lower_bounds.append(
                        min(expected_sum, constraint.get_sum_limit()))
                else:
                    cvar_lower_bounds.append(expected_sum)
                no_of_scenarios_to_consider =\
                    max(1, int(np.floor(
                        constraint.get_percentage_of_scenarios()\
                         *no_of_scenarios/100.0)))
                min_no_of_scenarios_to_consider.append(
                    no_of_scenarios_to_consider
                )
                max_no_of_scenarios_to_consider.append(
                    no_of_scenarios
                )
            else:
                cvarified_constraint = \
                    self.__get_cvarified_constraint(
                        constraint, constraint.get_sum_limit())

                cvar_threshold = \
                    self.__get_linearized_cvar_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        cvarified_constraint, no_of_scenarios
                    )
                cvar_upper_bounds.append(cvar_threshold)
                cvar_lower_bounds.append(
                    self.__get_expected_sum_among_optimization_scenarios(
                        probabilistically_unconstrained_package,
                        constraint.get_attribute_name()
                    )
                )
                no_of_scenarios_to_consider =\
                    max(1, int(np.floor(
                        cvarified_constraint.get_percentage_of_scenarios()\
                         *no_of_scenarios/100.0)))
                min_no_of_scenarios_to_consider.append(
                    no_of_scenarios_to_consider
                )
                max_no_of_scenarios_to_consider.append(
                    no_of_scenarios
                )
            stochastic_constraint_index += 1
        
        for idx in range(len(cvar_lower_bounds)):
            if cvar_lower_bounds[idx] > cvar_upper_bounds[idx]:
                cvar_upper_bounds[idx] -= cvar_upper_bounds[idx]
            else:
                cvar_upper_bounds[idx] += cvar_upper_bounds[idx]
        return cvar_upper_bounds, cvar_lower_bounds,\
            max_no_of_scenarios_to_consider,\
            min_no_of_scenarios_to_consider,\
            trivial_constraints
    

    def solve(self, can_add_scenarios = True):
        self.__metrics.start_execution()
        no_of_scenarios = self.__init_no_of_scenarios
        unacceptable_diff = True

        while unacceptable_diff:
            self.__model_setup(
                no_of_scenarios=no_of_scenarios,
                no_of_scenarios_to_consider=[],
                probabilistically_constrained=False
            )

            probabilistically_unconstrained_package = \
                self.__get_package()
            print('Probabilistically unconstrained package:',
                probabilistically_unconstrained_package)
            if probabilistically_unconstrained_package is None:
                self.__metrics.end_execution(0, 0)
                return (None, 0.0)
        
            probabilistically_unconstrained_package_with_indices = \
                self.__get_package_with_indices()

            unacceptable_diff, validation_objective_value = \
                self.__is_objective_value_relative_diff_high(
                    probabilistically_unconstrained_package,
                    probabilistically_unconstrained_package_with_indices,
                    no_of_scenarios
                )
            
            if not can_add_scenarios or not unacceptable_diff:
                objective_upper_bound = validation_objective_value
                break

            no_of_scenarios *= 2
        
        print('Objective value upper bound:', objective_upper_bound)

        if self.__validator.is_package_validation_feasible(
            probabilistically_unconstrained_package):
            print('Probabilistically unconstrained package'
                  ' is validation feasible')
            self.__metrics.end_execution(
                objective_upper_bound, 0)
            return (probabilistically_unconstrained_package,
                    objective_upper_bound)

        global_best_feasible_package = None
        global_best_feasible_objective = None

        while no_of_scenarios <= self.__max_opt_scenarios:

            cvar_upper_bounds, cvar_lower_bounds,\
            max_no_of_scenarios_to_consider,\
            min_no_of_scenarios_to_consider,\
            trivial_constraints = \
                self.__get_bounds_for_risk_constraints(
                    no_of_scenarios,
                    probabilistically_unconstrained_package,
                )

            self.__preprocess_lcvar_prefix_sums(no_of_scenarios)

            is_model_setup = False
            
            print('CVaR upper bounds:', cvar_upper_bounds)
            print('CVaR lower bounds:', cvar_lower_bounds)
            print('Min No of Scenarios to consider:',
                  min_no_of_scenarios_to_consider)
            print('Max no of scenarios to consider:',
                  max_no_of_scenarios_to_consider)
            print('Trivial constraints:', trivial_constraints)

            cvar_lower_bounds, is_model_setup, needs_more_scenarios, repair_feasible_package, repair_feasible_objective = \
                self.__repair_lower_bound_anchors(
                    no_of_scenarios=no_of_scenarios,
                    cvar_upper_bounds=cvar_upper_bounds,
                    cvar_lower_bounds=cvar_lower_bounds,
                    trivial_constraints=trivial_constraints,
                    no_of_scenarios_to_consider=\
                        min_no_of_scenarios_to_consider,
                    can_add_scenarios=can_add_scenarios
                )

            if repair_feasible_package is not None:
                if global_best_feasible_package is None or \
                    (self.__query.get_objective().get_objective_type() == ObjectiveType.MAXIMIZATION and repair_feasible_objective > global_best_feasible_objective) or \
                    (self.__query.get_objective().get_objective_type() != ObjectiveType.MAXIMIZATION and repair_feasible_objective < global_best_feasible_objective):
                    global_best_feasible_package = repair_feasible_package
                    global_best_feasible_objective = repair_feasible_objective

            if can_add_scenarios and needs_more_scenarios:
                return (None, 0.0, True)

            print('CVaR lower bounds after repair:', cvar_lower_bounds)

            init_diff = max(
                self.__l_inf(cvar_upper_bounds, cvar_lower_bounds),
                1e-9
            )

            while self.__l_inf(cvar_upper_bounds, cvar_lower_bounds)\
                >= self.__bisection_threshold*init_diff or\
                    self.__l_inf(min_no_of_scenarios_to_consider,
                                 max_no_of_scenarios_to_consider) >= 1:
                
                print('CVaR upper bounds:', cvar_upper_bounds)
                print('CVaR lower bounds:', cvar_lower_bounds)
                print('Min No of Scenarios to consider:',
                  min_no_of_scenarios_to_consider)
                print('Max no of scenarios to consider:',
                  max_no_of_scenarios_to_consider)
                print('Trivial constraints:', trivial_constraints)

                threshold_search_result = \
                    self.__cvar_threshold_search(
                        no_of_scenarios,
                        cvar_upper_bounds,
                        cvar_lower_bounds,
                        trivial_constraints,
                        objective_upper_bound,
                        min_no_of_scenarios_to_consider,
                        is_model_setup,
                        can_add_scenarios
                    )

                is_model_setup = True

                if can_add_scenarios and threshold_search_result.\
                    get_needs_more_scenarios():
                    break
                if threshold_search_result.\
                    get_found_appropriate_package():
                    self.__metrics.end_execution(
                        threshold_search_result.\
                            get_objective_value(),
                        no_of_scenarios
                    )
                    return (
                        threshold_search_result.get_package(),
                        threshold_search_result.get_objective_value()
                    )
                cvar_upper_bounds = \
                    threshold_search_result.get_cvar_thresholds()
                objective_upper_bound = \
                    threshold_search_result.get_objective_upper_bound()

                if threshold_search_result.get_best_feasible_package() is not None:
                    cand_pkg = threshold_search_result.get_best_feasible_package()
                    cand_obj = threshold_search_result.get_best_feasible_objective()
                    if global_best_feasible_package is None or \
                        (self.__query.get_objective().get_objective_type() == \
                            ObjectiveType.MAXIMIZATION and cand_obj > global_best_feasible_objective) or \
                        (self.__query.get_objective().get_objective_type() != \
                            ObjectiveType.MAXIMIZATION and cand_obj < global_best_feasible_objective):
                        global_best_feasible_package = cand_pkg
                        global_best_feasible_objective = cand_obj

                # Check if global best is within approximation bound of objective upper bound
                if global_best_feasible_package is not None and \
                        self.__is_objective_value_enough(global_best_feasible_objective, objective_upper_bound):
                    print('Best feasible package within approximation bound; returning.')
                    self.__metrics.end_execution(global_best_feasible_objective, no_of_scenarios)
                    return (global_best_feasible_package, global_best_feasible_objective)

                print('[Solve] Entering coefficient search'
                      ' | min scenarios:', min_no_of_scenarios_to_consider,
                      '| max scenarios:', max_no_of_scenarios_to_consider)
                coefficient_search_result = \
                    self.__cvar_coefficient_search(
                        no_of_scenarios,
                        cvar_upper_bounds,
                        trivial_constraints,
                        objective_upper_bound,
                        min_no_of_scenarios_to_consider,
                        max_no_of_scenarios_to_consider,
                        is_model_setup,
                        can_add_scenarios,
                    )
                print('[Solve] Coefficient search result'
                      ' | needs more scenarios:', coefficient_search_result.get_needs_more_scenarios(),
                      '| found package:', coefficient_search_result.get_found_appropriate_package())

                if coefficient_search_result.\
                    get_needs_more_scenarios():
                    break
                if coefficient_search_result.\
                    get_found_appropriate_package():
                    self.__metrics.end_execution(
                        coefficient_search_result.\
                            get_objective_value(),
                        no_of_scenarios
                    )
                    return (
                        coefficient_search_result.get_package(),
                        coefficient_search_result.get_objective_value()
                    )

                min_no_of_scenarios_to_consider = \
                    coefficient_search_result.get_scenarios_to_consider()
                objective_upper_bound = \
                    coefficient_search_result.get_objective_upper_bound()

                if coefficient_search_result.get_best_feasible_package() is not None:
                    cand_pkg = coefficient_search_result.get_best_feasible_package()
                    cand_obj = coefficient_search_result.get_best_feasible_objective()
                    if global_best_feasible_package is None or \
                        (self.__query.get_objective().get_objective_type() == \
                            ObjectiveType.MAXIMIZATION and cand_obj > global_best_feasible_objective) or \
                        (self.__query.get_objective().get_objective_type() != \
                            ObjectiveType.MAXIMIZATION and cand_obj < global_best_feasible_objective):
                        global_best_feasible_package = cand_pkg
                        global_best_feasible_objective = cand_obj

                # Check if global best is within approximation bound of objective upper bound
                if global_best_feasible_package is not None and \
                        self.__is_objective_value_enough(global_best_feasible_objective, objective_upper_bound):
                    print('Best feasible package within approximation bound; returning.')
                    self.__metrics.end_execution(global_best_feasible_objective, no_of_scenarios)
                    return (global_best_feasible_package, global_best_feasible_objective)

                # Convergence check: if no non-None infeasible packages found in this round
                if not (threshold_search_result.get_found_non_none_infeasible()
                        or coefficient_search_result.get_found_non_none_infeasible()):
                    if global_best_feasible_package is not None:
                        print('Both sub-searches found no infeasible packages; returning best feasible.')
                        self.__metrics.end_execution(global_best_feasible_objective, no_of_scenarios)
                        return (global_best_feasible_package, global_best_feasible_objective)
                    else:
                        print('No feasible package found at convergence.')
                        self.__metrics.end_execution(0, no_of_scenarios)
                        if self.__early_termination:
                            return (None, 0.0)

                for ind in range(len(cvar_upper_bounds)):
                    if cvar_upper_bounds[ind] <\
                        cvar_lower_bounds[ind]:
                        cvar_upper_bounds[ind] -= self.__bisection_threshold
                    else:
                        cvar_upper_bounds[ind] += self.__bisection_threshold

                for ind in range(len(min_no_of_scenarios_to_consider)):
                    if min_no_of_scenarios_to_consider[ind] <\
                        max_no_of_scenarios_to_consider[ind]:
                        max_no_of_scenarios_to_consider[ind] -= 1
            
            if no_of_scenarios == self.__max_opt_scenarios:
                print('Could not find a package within',
                      self.__max_opt_scenarios, 'scenarios')
                break
            if self.__early_termination and global_best_feasible_package is None:
                print('Early termination: no feasible package found, exiting.')
                self.__metrics.end_execution(0, no_of_scenarios)
                return (None, 0.0)
            no_of_scenarios *= 2
            if no_of_scenarios >= self.__max_opt_scenarios:
                no_of_scenarios = self.__max_opt_scenarios
        if global_best_feasible_package is not None:
            print('Returning best feasible package found (objective did not meet approximation ratio)')
            self.__metrics.end_execution(
                global_best_feasible_objective, no_of_scenarios)
            return (global_best_feasible_package, global_best_feasible_objective)
        self.__metrics.end_execution(0, no_of_scenarios)
        return (None, 0.0)
    

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

    def get_metrics(self) -> OptimizationMetrics:
        return self.__metrics
