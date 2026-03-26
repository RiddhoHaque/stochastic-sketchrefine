import numpy as np
import os
from numpy.random import SFC64, SeedSequence, Generator
from concurrent.futures import ThreadPoolExecutor
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator

SUBSTEPS = 10
DRIFT_DAMPEN = 0.2


def _hash(s: str) -> int:
    hashed_value = 0
    for char in s:
        hashed_value = hashed_value * 7727 + ord(char)
        hashed_value %= 2593697387
    return hashed_value


def _process_ticker(ticker, group, seed, no_of_scenarios):
    """
    group : list of (tuple_idx, sell_after, price, volatility, drift)
    All rows for the same ticker share price, volatility, and drift.
    """
    _, _, last_price, last_volatility, last_drift = group[0]

    sell_after_map = {int(sa): idx for idx, sa, *_ in group}
    sorted_dates   = sorted(sell_after_map.keys())

    hashed_value = (seed + _hash(ticker)) % (10 ** 8)
    rng = Generator(SFC64(SeedSequence(hashed_value)))

    drift_coeff = (last_drift - 0.5 * last_volatility ** 2) * DRIFT_DAMPEN

    # Build per-substep dt array, one interval of size 1 day per sell_after step
    intervals      = [d - prev_d for prev_d, d in zip([0] + sorted_dates[:-1], sorted_dates)]
    sub_step_sizes = [interval / SUBSTEPS for interval in intervals]

    expanded_dt = np.array(
        [sub_dt for sub_dt in sub_step_sizes for _ in range(SUBSTEPS)],
        dtype=np.float32
    )
    sqrt_dt     = np.sqrt(expanded_dt)
    total_steps = len(expanded_dt)

    extended_scenarios = int(no_of_scenarios * 1.2)
    Z = rng.standard_normal(size=(extended_scenarios, total_steps)).astype(np.float32)

    log_increments  = drift_coeff * expanded_dt + last_volatility * sqrt_dt * Z
    cum_log_prices  = np.cumsum(log_increments, axis=1)
    price_paths     = last_price * np.exp(cum_log_prices)

    # Trim bottom and top 10% by final price
    final_prices = price_paths[:, -1]
    low  = (extended_scenarios - no_of_scenarios) // 2
    high = low + no_of_scenarios
    sorted_indices = np.argsort(final_prices)[low:high]
    trimmed_paths  = price_paths[sorted_indices]

    extraction_indices = [SUBSTEPS * (i + 1) - 1 for i in range(len(sorted_dates))]

    rng.shuffle(trimmed_paths)
    return {
        sell_after_map[date]: (trimmed_paths[:, extraction_indices[i]] - last_price).tolist()
        for i, date in enumerate(sorted_dates)
    }


class GBMGainScenarioGenerator(ScenarioGenerator):

    def __init__(self, relation: str, base_predicate: str = ''):
        self.__relation = relation
        self.__base_predicate = base_predicate or '1=1'

    def __get_info(self):
        sql = (
            f'SELECT ticker, sell_after, price, volatility, drift '
            f'FROM {self.__relation} '
            f'WHERE {self.__base_predicate} ORDER BY id;'
        )
        PgConnection.Execute(sql)
        return PgConnection.Fetch()

    def generate_scenarios(self, seed: int, no_of_scenarios: int) -> list[list[float]]:
        info  = self.__get_info()
        gains: list[list[float]] = [[] for _ in range(len(info))]

        ticker_groups: dict[str, list] = {}
        for idx, row in enumerate(info):
            ticker, sell_after, price, volatility, drift = row
            ticker_groups.setdefault(ticker, []).append(
                (idx, sell_after, price, volatility, drift)
            )

        args = [
            (ticker, group, seed, no_of_scenarios)
            for ticker, group in ticker_groups.items()
        ]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(_process_ticker, *a) for a in args]
            for future in futures:
                try:
                    for tuple_idx, g in future.result().items():
                        gains[tuple_idx] = g
                except Exception as exc:
                    print(f'[GBMGainScenarioGenerator] Warning: ticker skipped — {exc}')

        return gains
