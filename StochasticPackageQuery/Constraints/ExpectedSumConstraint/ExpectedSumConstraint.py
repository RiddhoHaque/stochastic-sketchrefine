from StochasticPackageQuery.Constraints.DeterministicConstraint.DeterministicConstraint import DeterministicConstraint


class ExpectedSumConstraint(DeterministicConstraint):

    def __init__(self):
        super().__init__()

    def is_deterministic_constraint(self) -> bool:
        return False

    def is_expected_sum_constraint(self) -> bool:
        return True