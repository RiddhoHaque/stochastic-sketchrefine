import os
import numpy as np
from numpy.random import SFC64, SeedSequence, Generator
from concurrent.futures import ThreadPoolExecutor
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator


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

    for _ in range(no_of_scenarios):
        # Sample indices into residuals array for all steps at once
        z_indices   = rng.integers(0, n_res, size=max_steps)
        z_seq       = residuals[z_indices]

        h_t         = last_h
        cum_log_ret = 0.0
        step        = 0

        for date in sorted_dates:
            # Simulate from current step up to this sell_after checkpoint
            while step < date:
                z           = z_seq[step]
                r_t         = z * np.sqrt(h_t)
                h_t         = omega + alpha1 * r_t ** 2 + beta1 * h_t
                cum_log_ret += r_t
                step        += 1

            gain = price * (np.exp(cum_log_ret) - 1.0)
            tup_idx = sell_after_map[date]
            gains_by_idx.setdefault(tup_idx, []).append(gain)

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
