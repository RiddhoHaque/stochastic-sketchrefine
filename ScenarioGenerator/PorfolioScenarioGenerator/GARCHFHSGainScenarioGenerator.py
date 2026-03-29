import os
import numpy as np
from numpy.random import SFC64, SeedSequence, Generator
from concurrent.futures import ThreadPoolExecutor
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator
from Utils.Relation_Prefixes import Relation_Prefixes


def _hash(s: str) -> int:
    hashed_value = 0
    for char in s:
        hashed_value = hashed_value * 7727 + ord(char)
        hashed_value %= 2593697387
    return hashed_value


def _process_ticker(ticker, param_group, residuals, seed, no_of_scenarios):
    """
    param_group : list of (tuple_idx, sell_after, price, omega, alpha1, beta1, last_h, last_return)
    residuals   : 1-D numpy array of standardized residuals z_t for this ticker
    """
    # All parameter rows for the same ticker share the same GARCH params / price
    _, _, price, omega, alpha1, beta1, last_h, _ = param_group[0]

    sell_after_map = {int(sa): idx for idx, sa, *_ in param_group}
    sorted_dates   = sorted(sell_after_map.keys())
    max_steps      = sorted_dates[-1]

    hashed_value = (seed + _hash(ticker)) % (10 ** 8)
    rng = Generator(SFC64(SeedSequence(hashed_value)))

    n_res = len(residuals)
    gains_by_idx: dict[int, list[float]] = {}

    # Generate all residual indices for all scenarios at once: shape (S, T)
    z_indices = rng.integers(0, n_res, size=(no_of_scenarios, max_steps))
    z_mat     = residuals[z_indices]   # (S, T) — fancy index into residuals array

    # Vectorized GARCH simulation across S scenarios simultaneously
    h           = np.full(no_of_scenarios, last_h,  dtype=np.float64)  # (S,)
    cum_log_ret = np.zeros(no_of_scenarios,          dtype=np.float64)  # (S,)

    step = 0
    for date in sorted_dates:
        while step < date:
            z_col        = z_mat[:, step]                       # (S,)
            r_t          = z_col * np.sqrt(h)                   # (S,)
            h            = omega + alpha1 * r_t ** 2 + beta1 * h  # (S,)
            cum_log_ret += r_t                                   # (S,)
            step        += 1

        gains    = price * (np.exp(cum_log_ret) - 1.0)  # (S,)
        tup_idx  = sell_after_map[date]
        gains_by_idx[tup_idx] = gains.tolist()

    return gains_by_idx


class GARCHFHSGainScenarioGenerator(ScenarioGenerator):

    def __init__(self, relation: str, residuals_relation: str = None,
                 base_predicate: str = ''):
        self.__relation = relation
        # Default: replace 'Portfolio' with 'Residuals' in the relation name
        self.__residuals_relation = (
            residuals_relation
            if residuals_relation is not None
            else relation.replace('Portfolio', 'Residuals')
        )
        self.__base_predicate = base_predicate or '1=1'

    def __get_params(self):
        sql = (
            f'SELECT ticker, sell_after, price, omega, alpha1, beta1, last_h, last_return '
            f'FROM {self.__relation} '
            f'WHERE {self.__base_predicate} '
            f'ORDER BY id;'
        )
        PgConnection.Execute(sql)
        return PgConnection.Fetch()

    def __get_residuals(self):
        sql = (
            f'SELECT ticker, residual_value '
            f'FROM {self.__residuals_relation} '
            f'ORDER BY ticker, id;'
        )
        PgConnection.Execute(sql)
        rows = PgConnection.Fetch()
        residuals: dict[str, list[float]] = {}
        for ticker, z in rows:
            residuals.setdefault(ticker, []).append(z)
        return {t: np.array(zs, dtype=np.float64) for t, zs in residuals.items()}

    def generate_scenarios(self, seed: int, no_of_scenarios: int) -> list[list[float]]:
        param_rows = self.__get_params()
        residuals  = self.__get_residuals()

        gains: list[list[float]] = [[] for _ in range(len(param_rows))]

        # Group parameter rows by ticker, preserving original index for output ordering
        ticker_groups: dict[str, list] = {}
        for idx, row in enumerate(param_rows):
            ticker, sell_after, price, omega, alpha1, beta1, last_h, last_return = row
            ticker_groups.setdefault(ticker, []).append(
                (idx, sell_after, price, omega, alpha1, beta1, last_h, last_return)
            )

        args = [
            (ticker, group, residuals[ticker], seed, no_of_scenarios)
            for ticker, group in ticker_groups.items()
            if ticker in residuals and len(residuals[ticker]) > 0
        ]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(_process_ticker, *a) for a in args]
            for future in futures:
                try:
                    for tuple_idx, g in future.result().items():
                        gains[tuple_idx] = g
                except Exception as exc:
                    print(f'[GARCHFHSGainScenarioGenerator] Warning: ticker skipped — {exc}')

        return gains

    def generate_scenarios_from_partition(
        self, seed: int, no_of_scenarios: int,
        partition_id: int
    ) -> list[list[float]]:
        self.__relation = self.__relation + \
            ' AS r INNER JOIN ' + \
            Relation_Prefixes.PARTITION_RELATION_PREFIX + \
            self.__relation + ' AS p ON r.id=p.tuple_id'

        if len(self.__base_predicate) > 0:
            self.__base_predicate += ' AND '
        self.__base_predicate += 'p.partition_id = ' + str(partition_id)

        return self.generate_scenarios(seed, no_of_scenarios)
