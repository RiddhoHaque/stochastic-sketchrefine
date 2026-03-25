import numpy as np

from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGenerator import RepresentativeScenarioGenerator
from ScenarioGenerator.RepresentativeScenarioGenerator.RepresentativeScenarioGeneratorWithoutCorrelation import RepresentativeScenarioGeneratorWithoutCorrelation
from SketchRefine.RefineValidator import RefineValidator
from SketchRefine.RefineRCLSolve import RefineRCLSolve
from StochasticPackageQuery.Query import Query
from Utils.Stochasticity import Stochasticity
from Utils.Relation_Prefixes import Relation_Prefixes
from ValueGenerator.RepresentativeValueGenerator import RepresentativeValueGenerator
from ValueGenerator.ValueGenerator import ValueGenerator



class Refine:

    def __init__(
        self, partition_groups: list[list[int]],
        no_of_optimization_scenarios: int,
        max_no_of_duplicates: list[int],
        sketch_objective_value: float,
        sketch_package: dict[int, int],
        query: Query, dbInfo: DbInfo,
        linear_relaxation: bool
    ):

        self.__partition_groups = partition_groups
        self.__no_of_optimization_scenarios =\
            no_of_optimization_scenarios
        self.__no_of_validation_scenarios =\
            Hyperparameters.NO_OF_VALIDATION_SCENARIOS
        self.__max_no_of_duplicates =\
            max_no_of_duplicates
        self.__sketch_objective_value =\
            sketch_objective_value
        self.__sketch_package =\
            sketch_package
        
        self.__query = query
        self.__tuples_in_partition =\
            dict()
        self.get_tuples_in_each_partition()
        
        self.__representatives =\
            dict()
        self.get_representative_of_each_partition()

        self.__stochastic_attributes = \
            self.__get_stochastic_attributes()
        self.__deterministic_attributes = \
            self.__get_deterministic_attributes()
        self.__dbInfo = dbInfo

        self.__partition_optimization_scenarios = dict()
        self.__partition_validation_scenarios = dict()
        self.__partition_values = dict()

        self.__chosen_tuple_optimization_scenarios = dict()
        self.__chosen_tuple_validation_scenarios = dict()
        self.__chosen_tuple_values = dict()

        self.__partition_variable_multiplicity = dict()
        self.__chosen_tuple_multiplicity = dict()
        self.__total_optimizer_runtime = 0.0
        self.__total_optimization_calls = 0

        for attr in self.__deterministic_attributes:
            self.__partition_values[attr] = dict()
            self.__chosen_tuple_values[attr] = dict()

            for partition_group in self.__partition_groups:
                for partition in partition_group:
                    num_duplicates = int(self.__sketch_package[partition])
                    if len(self.__tuples_in_partition[partition]) < num_duplicates:
                        num_duplicates = int(len(self.__tuples_in_partition[partition]))
                    self.__partition_values[attr][partition] = \
                        RepresentativeValueGenerator(
                            relation=self.__query.get_relation(),
                            base_predicate='r.partition_id=' + str(partition),
                            attribute=attr,
                            duplicate_vector=[num_duplicates]
                        ).get_values()


        for attr in self.__stochastic_attributes:
            
            self.__partition_optimization_scenarios[attr] = dict()
            self.__partition_validation_scenarios[attr] = dict()

            self.__chosen_tuple_optimization_scenarios[attr] = dict()
            self.__chosen_tuple_validation_scenarios[attr] = dict()

            for partition_group in self.__partition_groups:
                for partition in partition_group:
                    
                    num_duplicates = int(self.__sketch_package[partition])
                    if len(self.__tuples_in_partition[partition]) < num_duplicates:
                        num_duplicates = len(self.__tuples_in_partition[partition])

                    self.__partition_optimization_scenarios[attr][partition] = \
                        RepresentativeScenarioGeneratorWithoutCorrelation(
                            relation=self.__query.get_relation(),
                            attr=attr, 
                            scenario_generator=\
                                self.__dbInfo.get_variable_generator_function(attr),
                            base_predicate='partition_id=' + str(partition),
                            duplicates=[num_duplicates]
                        ).generate_scenarios(
                            seed=Hyperparameters.INIT_SEED,
                            no_of_scenarios=no_of_optimization_scenarios
                        )
                    
                    if self.__dbInfo.has_inter_tuple_correlations():
                        self.__partition_validation_scenarios[attr][partition] = \
                            RepresentativeScenarioGenerator(
                                relation=self.__query.get_relation(),
                                attr=attr,
                                dbInfo=self.__dbInfo
                            ).generate_scenarios(
                                seed=Hyperparameters.VALIDATION_SEED,
                                no_of_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
                                pid=partition,
                                duplicates_to_use=num_duplicates
                            )
                    else:
                        self.__partition_validation_scenarios[attr][partition] = \
                            RepresentativeScenarioGeneratorWithoutCorrelation(
                                relation=self.__query.get_relation(),
                                attr=attr, 
                                scenario_generator=\
                                    self.__dbInfo.get_variable_generator_function(attr),
                                base_predicate='partition_id=' + str(partition),
                                duplicates=[num_duplicates]
                            ).generate_scenarios(
                                seed=Hyperparameters.VALIDATION_SEED,
                                no_of_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS
                            )
                                        
        for partition_group in self.__partition_groups:
            for partition in partition_group:
                num_duplicates = int(self.__sketch_package[partition])
                if len(self.__tuples_in_partition[partition]) < num_duplicates:
                    num_duplicates = len(self.__tuples_in_partition[partition])
                self.__partition_variable_multiplicity[partition] = [
                    int(self.__sketch_package[partition]) // num_duplicates\
                        for _ in range(num_duplicates)
                ]
                for _ in range(int(self.__sketch_package[partition]) % num_duplicates):
                    self.__partition_variable_multiplicity[partition][_] += 1
        self.__is_linear_relaxation = linear_relaxation

    def get_tuples_in_each_partition(self):
        partition_predicate = ''
        for partition_group in self.__partition_groups:
            for partition_id in partition_group:
                if len(partition_predicate) > 0:
                    partition_predicate += ' or '
                partition_predicate += 'p.partition_id = ' +\
                    str(partition_id)

        relation = self.__query.get_relation()
        partition_relation = Relation_Prefixes.PARTITION_RELATION_PREFIX + relation
        query_base = self.__query.get_base_predicate()

        if query_base and query_base != '1=1':
            sql = (
                f'SELECT p.partition_id, p.tuple_id FROM {relation} AS r '
                f'INNER JOIN {partition_relation} AS p ON r.id = p.tuple_id '
                f'WHERE ({partition_predicate}) AND ({query_base}) '
                f'ORDER BY (p.partition_id, p.tuple_id);'
            )
        else:
            sql = (
                f'SELECT p.partition_id, p.tuple_id FROM {partition_relation} AS p '
                f'WHERE {partition_predicate} '
                f'ORDER BY (p.partition_id, p.tuple_id);'
            )

        PgConnection.Execute(sql)

        tuples = PgConnection.Fetch()

        for pid, tid in tuples:
            if pid not in self.__tuples_in_partition:
                self.__tuples_in_partition[pid] = []
            self.__tuples_in_partition[pid].append(tid)


    def get_representative_of_each_partition(self):
        base_predicate = ''
        for partition_group in self.__partition_groups:
            for partition_id in partition_group:
                if len(base_predicate) > 0:
                    base_predicate += ' or '
                base_predicate += 'partition_id = ' +\
                    str(partition_id)
        
        sql = 'SELECT partition_id, attribute, representative_tuple_id ' +\
            'FROM ' + Relation_Prefixes.REPRESENTATIVE_RELATION_PREFIX +\
            self.__query.get_relation() + ' WHERE ' + base_predicate + ';'
        
        PgConnection.Execute(sql)
        tuples = PgConnection.Fetch()

        for partition_id, attribute, representative_tuple_id in tuples:
            self.__representatives[(partition_id, attribute)] =\
                representative_tuple_id


    def __get_stochastic_attributes(self):
        attributes = set()
        for constraint in self.__query.get_constraints():
            if constraint.is_expected_sum_constraint():
                attributes.add(
                    constraint.get_attribute_name())
            if constraint.is_risk_constraint():
                attributes.add(
                    constraint.get_attribute_name())
        
        if self.__query.get_objective().get_stochasticity()\
            == Stochasticity.STOCHASTIC:
            attributes.add(self.__query.get_objective().\
                           get_attribute_name())
        
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
                self.__query.get_objective().\
                    get_attribute_name())
        return attributes


    def get_optimizer_runtime(self):
        return self.__total_optimizer_runtime

    def get_number_of_optimization_calls(self):
        return self.__total_optimization_calls

    def solve(self):
        forward_bins = []
        self.__partition_groups.reverse()

        for partition_group in self.__partition_groups:
            forward_bins.append(partition_group)

        chosen_tuples_per_bin = []

        group_index = 0

        while group_index < len(forward_bins):
            
            package, objective_value = \
                self.solve_partition(
                    forward_bins[group_index],
                )
            
            if package is None:
                if group_index > 0:
                    forward_bins[group_index], forward_bins[group_index-1] =\
                        forward_bins[group_index-1], forward_bins[group_index]
                    
                    for chosen_tuple in chosen_tuples_per_bin[-1]:
                        del self.__chosen_tuple_multiplicity[chosen_tuple]
                    
                    for attr in self.__deterministic_attributes:
                        for chosen_tuple in chosen_tuples_per_bin[-1]:
                            del self.__chosen_tuple_values[attr][chosen_tuple]

                        for partition in forward_bins[group_index]:

                            num_duplicates = int(self.__sketch_package[partition])
                            if len(self.__tuples_in_partition[partition]) < num_duplicates:
                                num_duplicates = len(self.__tuples_in_partition[partition])
                            
                            self.__partition_values[attr][partition] = \
                                RepresentativeValueGenerator(
                                    relation=self.__query.get_relation(),
                                    base_predicate='r.partition_id=' + str(partition),
                                    attribute=attr,
                                    duplicate_vector=[num_duplicates]
                                ).get_values()
                    
                    for attr in self.__stochastic_attributes:
                        for chosen_tuple in chosen_tuples_per_bin[-1]:
                            del self.__chosen_tuple_optimization_scenarios[attr][chosen_tuple]
                            del self.__chosen_tuple_validation_scenarios[attr][chosen_tuple]
                        
                        for partition in forward_bins[group_index]:

                            num_duplicates = int(self.__sketch_package[partition])
                            if len(self.__tuples_in_partition[partition]) < num_duplicates:
                                num_duplicates = len(self.__tuples_in_partition[partition])

                            self.__partition_optimization_scenarios[attr][partition] = \
                                RepresentativeScenarioGeneratorWithoutCorrelation(
                                    relation=self.__query.get_relation(),
                                    attr=attr, 
                                    scenario_generator=\
                                        self.__dbInfo.get_variable_generator_function(attr),
                                    base_predicate='partition_id=' + str(partition),
                                    duplicates=[num_duplicates]
                                ).generate_scenarios(
                                    seed=Hyperparameters.INIT_SEED,
                                    no_of_scenarios=self.__no_of_optimization_scenarios
                                )

                            if self.__dbInfo.has_inter_tuple_correlations():
                                self.__partition_validation_scenarios[attr][partition] = \
                                    RepresentativeScenarioGenerator(
                                        relation=self.__query.get_relation(),
                                        attr=attr,
                                        dbInfo=self.__dbInfo
                                    ).generate_scenarios(
                                        seed=Hyperparameters.VALIDATION_SEED,
                                        no_of_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
                                        pid=partition,
                                        duplicates_to_use=num_duplicates
                                    )
                            else:
                                self.__partition_validation_scenarios[attr][partition] = \
                                    RepresentativeScenarioGeneratorWithoutCorrelation(
                                        relation=self.__query.get_relation(),
                                        attr=attr, 
                                        scenario_generator=\
                                            self.__dbInfo.get_variable_generator_function(attr),
                                        base_predicate='partition_id=' + str(partition),
                                        duplicates=[num_duplicates]
                                    ).generate_scenarios(
                                        seed=Hyperparameters.VALIDATION_SEED,
                                        no_of_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS
                                    )

                    for partition in forward_bins[group_index]:
                        num_duplicates = int(self.__sketch_package[partition])
                        if len(self.__tuples_in_partition[partition]) < num_duplicates:
                            num_duplicates = len(self.__tuples_in_partition[partition])
                        self.__partition_variable_multiplicity[partition] = [
                            int(self.__sketch_package[partition]) // num_duplicates
                            for _ in range(num_duplicates)
                        ]
                        for _ in range(int(self.__sketch_package[partition]) % num_duplicates):
                            self.__partition_variable_multiplicity[partition][_] += 1

                    chosen_tuples_per_bin.pop()
                    group_index -= 1
                else:
                    print('Sketch package could not be refined')
                    return None, 0.0
            
            else:
                chosen_tuples_per_bin.append([])
                for tuple_id in package:
                    chosen_tuples_per_bin[-1].append(tuple_id)
                group_index += 1
                
        refined_package = dict()

        for chosen_tuple in self.__chosen_tuple_multiplicity:
            refined_package[chosen_tuple] = \
                self.__chosen_tuple_multiplicity[chosen_tuple]
        
        return refined_package, objective_value


    def solve_partition(
        self, 
        partition_group
    ):

        chosen_partition_optimization_scenarios = dict()

        for attr in self.__stochastic_attributes:
            chosen_partition_optimization_scenarios[attr] = []
            
            for partition_id in partition_group:
                optimization_scenarios_for_partition = \
                    self.__dbInfo.get_variable_generator_function(attr)(
                        relation=self.__query.get_relation(),
                        base_predicate=self.__query.get_base_predicate()
                    ).generate_scenarios_from_partition(
                        seed=Hyperparameters.INIT_SEED,
                        no_of_scenarios=self.__no_of_optimization_scenarios,
                        partition_id=partition_id
                    )
                for scenario in optimization_scenarios_for_partition:
                    chosen_partition_optimization_scenarios[attr].append(scenario)
            
        chosen_partition_values = dict()
        for attr in self.__deterministic_attributes:
            chosen_partition_values[attr] = []
            
            for partition_id in partition_group:
                values = ValueGenerator(
                    relation=self.__query.get_relation(),
                    base_predicate=self.__query.get_base_predicate(),
                    attribute=attr
                ).get_values_from_partition(partition_id)

                for value in values:
                    chosen_partition_values[attr].append(value)
        
        partition_group_set = set(partition_group)

        filtered_partition_opt_scenarios = {
            attr: {
                p: self.__partition_optimization_scenarios[attr][p]
                for p in self.__partition_optimization_scenarios[attr]
                if p not in partition_group_set
            }
            for attr in self.__stochastic_attributes
        }

        filtered_partition_val_scenarios = {
            attr: {
                p: self.__partition_validation_scenarios[attr][p]
                for p in self.__partition_validation_scenarios[attr]
                if p not in partition_group_set
            }
            for attr in self.__stochastic_attributes
        }

        filtered_partition_values = {
            attr: {
                p: self.__partition_values[attr][p]
                for p in self.__partition_values[attr]
                if p not in partition_group_set
            }
            for attr in self.__deterministic_attributes
        }

        filtered_partition_var_multiplicity = {
            p: self.__partition_variable_multiplicity[p]
            for p in self.__partition_variable_multiplicity
            if p not in partition_group_set
        }

        validator = RefineValidator(
            query=self.__query,
            dbInfo=self.__dbInfo,
            no_of_validation_scenarios=self.__no_of_validation_scenarios,
            partition_validation_scenarios=filtered_partition_val_scenarios,
            chosen_tuple_validation_scenarios=self.__chosen_tuple_validation_scenarios,
            partition_variable_multiplicities=filtered_partition_var_multiplicity,
            chosen_tuple_multiplicities=self.__chosen_tuple_multiplicity
        )

        tuple_ids = []
        for partition in partition_group:
            for id in self.__tuples_in_partition[partition]:
                tuple_ids.append(id)

        refineRCLSolve = RefineRCLSolve(
            query=self.__query,
            linear_relaxation=self.__is_linear_relaxation,
            chosen_partition_optimization_scenarios=\
                chosen_partition_optimization_scenarios,
            chosen_partition_values=\
                chosen_partition_values,
            partition_optimization_scenarios=\
                filtered_partition_opt_scenarios,
            partition_values=\
                filtered_partition_values,
            chosen_tuple_optimization_scenarios=\
                self.__chosen_tuple_optimization_scenarios,
            chosen_tuple_values=\
                self.__chosen_tuple_values,
            validator=validator,
            approximation_bound=\
                Hyperparameters.APPROXIMATION_BOUND,
            bisection_threshold=\
                Hyperparameters.BISECTION_THRESHOLD,
            partition_variable_multiplicity=\
                filtered_partition_var_multiplicity,
            chosen_tuple_multiplicity=\
                self.__chosen_tuple_multiplicity,
            tuple_ids=tuple_ids,
            sketch_objective_value=\
                self.__sketch_objective_value
        )

        refined_package, objective_value,\
            needs_more_scenarios = refineRCLSolve.solve()
        m = refineRCLSolve.get_metrics()
        self.__total_optimizer_runtime += m.get_optimizer_runtime()
        self.__total_optimization_calls += m.get_number_of_optimization_calls()

        if refined_package is None:
            return refined_package, 0.0
        
        for partition_id in partition_group:
            for attr in self.__stochastic_attributes:
                del self.__partition_optimization_scenarios[attr][partition_id]
                del self.__partition_validation_scenarios[attr][partition_id]
            
            for attr in self.__deterministic_attributes:
                del self.__partition_values[attr][partition_id]

            del self.__partition_variable_multiplicity[partition_id]
        
        for tuple_id in refined_package:
            self.__chosen_tuple_multiplicity[tuple_id] = \
                int(refined_package[tuple_id])
            
            for attr in self.__stochastic_attributes:
                self.__chosen_tuple_optimization_scenarios[attr][tuple_id] =\
                    self.__dbInfo.get_variable_generator_function(
                        attr)(
                            relation=self.__query.get_relation(),
                            base_predicate=' id=' + str(tuple_id)
                        ).generate_scenarios(
                            seed=Hyperparameters.INIT_SEED,
                            no_of_scenarios = self.__no_of_optimization_scenarios,
                        )
                
                self.__chosen_tuple_validation_scenarios[attr][tuple_id] =\
                    self.__dbInfo.get_variable_generator_function(
                        attr)(
                            relation=self.__query.get_relation(),
                            base_predicate=' id=' + str(tuple_id)
                        ).generate_scenarios(
                            seed=Hyperparameters.VALIDATION_SEED,
                            no_of_scenarios=Hyperparameters.NO_OF_VALIDATION_SCENARIOS
                        )
            
            for attr in self.__deterministic_attributes:
                self.__chosen_tuple_values[attr][tuple_id] =\
                    ValueGenerator(
                        relation=self.__query.get_relation(),
                        base_predicate=' id=' + str(tuple_id),
                        attribute=attr
                    ).get_values()
            
        return refined_package, objective_value
