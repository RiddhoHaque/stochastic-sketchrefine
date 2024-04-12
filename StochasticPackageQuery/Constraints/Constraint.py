class Constraint:

    def is_repeat_constraint(self) -> bool:
        return False

    def is_package_size_constraint(self) -> bool:
        return False

    def is_deterministic_constraint(self) -> bool:
        return False

    def is_expected_sum_constraint(self) -> bool:
        return False

    def is_risk_constraint(self) -> bool:
        return False

    def is_var_constraint(self) -> bool:
        return False

    def is_cvar_constraint(self) -> bool:
        return False