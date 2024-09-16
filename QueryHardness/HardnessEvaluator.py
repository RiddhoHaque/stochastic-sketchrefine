import random

from CVaRification.QuickSolve import QuickSolve
from CVaRification.RCLSolve import RCLSolve
from DbInfo.DbInfo import DbInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from Validator.Validator import Validator
from ValueGenerator.ValueGenerator import ValueGenerator
from StochasticPackageQuery.Query import Query
from Utils.RelationalOperators import RelationalOperators
from Utils.Stochasticity import Stochasticity


class HardnessEvaluator:

    def __init__(
        self,
        query: Query,
        dbInfo: DbInfo,
        solver: RCLSolve,
        validator: Validator,
        foraging_period= 10
    ):
        self.__query = query
        self.__dbInfo = dbInfo
        self.__quick_solver = QuickSolve(
            query=self.__query,
            dbInfo= self.__dbInfo,
            init_no_of_scenarios=\
                Hyperparameters.INIT_NO_OF_SCENARIOS,
            no_of_validation_scenarios=\
                Hyperparameters.NO_OF_VALIDATION_SCENARIOS,
            sampling_tolerance=\
                Hyperparameters.SAMPLING_TOLERANCE
        )
        self.__quick_solved_package, self.__upper_bound,\
            self.__quick_solved = self.__quick_solver.solve()
        self.__solver = solver
        self.__init_package, _ = \
            self.__solver.solve()
        print('Init Package:', self.__init_package)
        
        self.__package_size_upper_bound = None
        self.__package_size_lower_bound = None
        for constraint in self.__query.get_constraints():
            if constraint.is_package_size_constraint():
                if constraint.get_inequality_sign() ==\
                    RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                    self.__package_size_lower_bound =\
                        constraint.get_package_size_limit()
                elif constraint.get_inequality_sign() ==\
                    RelationalOperators.LESS_THAN_OR_EQUAL_TO:
                    self.__package_size_upper_bound =\
                        constraint.get_package_size_limit()
                else:
                    self.__package_size_lower_bound =\
                        constraint.get_package_size_limit()
                    self.__package_size_upper_bound =\
                        constraint.get_package_size_limit()

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

        self.__validator = \
            validator
        self.__acceptable_packages = set()
        self.__ids = \
            ValueGenerator(
                relation=\
                    self.__query.get_relation(),
                base_predicate=\
                    self.__query.get_base_predicate(),
                attribute='id'
            ).get_values()
        self.__id_index = dict()
        iter = 0
        for id in self.__ids:
            self.__id_index[id[0]] = iter
            iter += 1
        self.__foraging_period =\
            foraging_period


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
                    get_attribute_name()
            )
        return attributes
    

    def satisfies_deterministic_constraints(
        self, package) -> bool:
        package_with_indices = dict()
        for tuple in package:
            package_with_indices[
                self.__id_index[tuple]
            ] = package[tuple]

        for constraint in self.__query.get_constraints():
            if constraint.is_deterministic_constraint():
                attr = constraint.get_attribute_name()
                sum = 0
                for index in package_with_indices:
                    sum += self.__values[attr][index]*\
                        package_with_indices[index]
                if constraint.get_inequality_sign() ==\
                    RelationalOperators.EQUALS:
                    if sum != constraint.get_sum_limit():
                        return False
                elif constraint.get_inequality_sign() ==\
                    RelationalOperators.GREATER_THAN_OR_EQUAL_TO:
                    if sum < constraint.get_sum_limit():
                        return False
                else:
                    if sum > constraint.get_sum_limit():
                        return False
        return True


    def __dot(
        self,
        direction_1: dict[int, int],
        direction_2: dict[int, int],
    ) -> int:
        dot = 0
        for id in direction_1:
            if id in direction_2:
                dot += direction_1[id] *\
                    direction_2[id]
        return dot


    def __has_negative_dot_product(
        self,
        step_direction: dict[int, int],
        step_history: list[dict[int, int]]
    ) -> bool:
        for direction in step_history:
            if self.__dot(
                step_direction,direction) < 0:
                return True
        return False
    

    def step(
        self, package: dict[int, int],
        step_history: list[dict[int, int]]
    ) -> tuple[dict[int, int], dict[int, int]]:
        options = []
        for id in package:
            if package[id] == 0:
                continue
            # Option 1: Delete the tuple
            new_package = dict()
            step_direction = dict()
            for _ in package:
                if package[_] > 0:
                    new_package[_] = \
                        package[_]
            new_package[id] -= 1
            step_direction[id] = -1
            if not self.__has_negative_dot_product(
                step_direction, step_history
            ):
                options.append(
                    (new_package, step_direction))

            # Option 2: Replace the tuple
            new_package = dict()
            step_direction = dict()
            for _ in package:
                if package[_] > 0:
                    new_package[_] = package[_]
            new_package[id] -= 1
            step_direction[id] = -1
            replacing_id = id
            while replacing_id == id:
                replacing_id = \
                    random.choice(self.__ids)[0]
            if replacing_id not in new_package:
                new_package[replacing_id] = 1
            else:
                new_package[replacing_id] += 1
            step_direction[replacing_id] = 1
            if not self.__has_negative_dot_product(
                step_direction, step_history
            ):
                options.append(
                    (new_package, step_direction))

        # Option 3: Insert a new tuple
        new_package = dict()
        step_direction = dict()
        for _ in package:
            new_package[_] = package[_]
        inserted_id = random.choice(
            self.__ids)[0]
        if inserted_id not in new_package:
            new_package[inserted_id] = 1
        else:
            new_package[inserted_id] += 1
        step_direction[inserted_id] = 1
        options.append(
            (new_package, step_direction))

        if len(options) == 0:
            return (package, dict())
        return random.choice(options)
    
    def make_hashable(self, o):
        t = type(o)
        if isinstance(o, dict):
            o = frozenset((k, self.make_hashable(v)) for k, v in o.items())
        elif isinstance(o, list):
            o = tuple(self.make_hashable(elem) for elem in o)
        elif isinstance(o, set):
            o = frozenset(self.make_hashable(elem) for elem in o)
        return t, o


    def within_package_size_limits(
        self, package: dict
    ) -> bool:
        if self.__package_size_lower_bound is None and\
            self.__package_size_upper_bound is None:
            return True
        if self.__package_size_lower_bound is None:
            return len(package) <= self.__package_size_upper_bound
        if self.__package_size_upper_bound is None:
            return len(package) >= self.__package_size_lower_bound
        return (len(package) >= self.__package_size_lower_bound and\
                    len(package) <= self.__package_size_upper_bound)


    def random_walk(self) -> int:
        step_count = 0
        current_package = \
            self.__init_package
        current_foraging_length = 0
        step_history = []
        while current_foraging_length < \
            self.__foraging_period:
            next_package, step_direction = \
                self.step(
                    current_package,
                    step_history)
            if next_package == current_package:
                return step_count
            current_package = next_package
            step_history.append(step_direction)
            if self.within_package_size_limits(current_package) and\
                self.satisfies_deterministic_constraints(current_package) and\
                self.__validator.is_package_validation_feasible(
                    current_package
                ):
                self.__acceptable_packages.add(
                    self.make_hashable(
                        current_package))
                step_count += 1
            
            else:
                current_foraging_length += 1

        return step_count


    def get_hardness(
        self, no_of_random_walks: int
    ) -> float:

        if self.__quick_solved:
            if self.__quick_solved_package is None:
                print('Probabilistically unconstrained'
                      ' query is unsolvable')
            else:
                print('Probabilistically unconstrained'
                      ' package is feasible')
            return -1

        step_count = 0
        self.__acceptable_packages.add(
            self.make_hashable(
                self.__init_package
            )
        )
        for _ in range(no_of_random_walks):
            step_count += \
                self.random_walk()
            if _ % 2000 == 0:
                print('Completed', _, 'walks')

        return len(self.__acceptable_packages)
        # return float(step_count)/\
        #    no_of_random_walks
    
    def log_hardness(
        self, query_no: int,
        db_name: str
    ):
        hardness = self.get_hardness(
            Hyperparameters.NO_OF_RANDOM_WALKS
        )
        f = open(db_name+'Q'+str(query_no) +'Hardness.txt', "w")
        f.write(str(hardness))
        f.close()
