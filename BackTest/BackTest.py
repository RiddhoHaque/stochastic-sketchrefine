import re

import pandas as pd
import pandas_market_calendars as mcal

from PgConnection.PgConnection import PgConnection


# ---------------------------------------------------------------------------
# NYSE calendar helper
# ---------------------------------------------------------------------------

def _build_sell_after_dates(year: int) -> dict:
    """
    Return dict  sell_after_int -> pd.Timestamp  for every NYSE trading day
    in the window  July 1, *year*  ..  June 30, *year*+1.
    sell_after == 1 is the first trading day on or after July 1.
    """
    nyse  = mcal.get_calendar('NYSE')
    start = pd.Timestamp(year=year,     month=7, day=1)
    end   = pd.Timestamp(year=year + 1, month=6, day=30)
    sched = nyse.schedule(start_date=start, end_date=end)
    dates = sched.index.normalize()
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    return {i + 1: dates[i] for i in range(len(dates))}


# ---------------------------------------------------------------------------
# BackTest
# ---------------------------------------------------------------------------

class BackTest:
    """
    Compute the actual gain a package would have produced given a relation
    drawn from GBM_Portfolio_YYYY or GARCH_Portfolio_YYYY.

    Usage
    -----
    bt = BackTest('GBM_Portfolio_2020')
    gain = bt.get_gain({42: 1.0, 107: 2.0})   # tuple IDs -> multiplicities
    """

    def __init__(self, relation_name: str):
        """
        Parameters
        ----------
        relation_name : str
            Name of a GBM_Portfolio_YYYY or GARCH_Portfolio_YYYY table.
            The year is extracted from the trailing four digits.
        """
        self.__relation   = relation_name
        self.__year       = self.__parse_year(relation_name)
        self.__sa_to_date = _build_sell_after_dates(self.__year)
        self.__ensure_cache_table()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def __ensure_cache_table():
        PgConnection.Execute("""
            CREATE TABLE IF NOT EXISTS ActualClose (
                ticker  varchar(10)      NOT NULL,
                date    date             NOT NULL,
                close   double precision NOT NULL,
                PRIMARY KEY (ticker, date)
            )
        """)
        PgConnection.Commit()

    @staticmethod
    def __parse_year(relation_name: str) -> int:
        match = re.search(r'(\d{4})$', relation_name)
        if not match:
            raise ValueError(
                f'Cannot parse a 4-digit year from relation name: {relation_name!r}'
            )
        return int(match.group(1))

    def __fetch_tuple_info(self, tuple_ids: list) -> dict:
        """
        Query the relation for (ticker, sell_after, price) for each id.

        Returns
        -------
        dict  tuple_id -> (ticker: str, sell_after: int, price: float)
        """
        if not tuple_ids:
            return {}

        ids_array = '{' + ','.join(str(int(tid)) for tid in tuple_ids) + '}'
        sql       = (
            f'SELECT id, ticker, sell_after, price '
            f'FROM {self.__relation} '
            f"WHERE id = ANY('{ids_array}'::int[]);"
        )
        PgConnection.Execute(sql)
        rows = PgConnection.Fetch()

        return {
            int(row[0]): (str(row[1]), int(row[2]), float(row[3]))
            for row in rows
        }

    def __fetch_closes(self, tickers: list) -> dict:
        """
        Load adjusted close prices for *tickers* over the relation's year window
        from the ActualClose cache table.

        Returns
        -------
        dict  ticker -> pd.Series  (Timestamp-indexed, ascending, tz-naive)
        """
        if not tickers:
            return {}

        start        = pd.Timestamp(year=self.__year,     month=7, day=1).date()
        end          = pd.Timestamp(year=self.__year + 1, month=6, day=30).date()
        ticks_array  = '{' + ','.join(t.replace('"', '') for t in tickers) + '}'
        sql          = (
            f"SELECT ticker, date, close FROM ActualClose "
            f"WHERE ticker = ANY('{ticks_array}'::varchar[]) "
            f"AND date BETWEEN '{start}' AND '{end}' "
            f"ORDER BY ticker, date;"
        )
        PgConnection.Execute(sql)
        rows = PgConnection.Fetch()

        data: dict = {}
        for ticker, date, close in rows:
            data.setdefault(ticker, {})[pd.Timestamp(date)] = close
        return {t: pd.Series(v) for t, v in data.items()}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_gain(self, package_dict: dict) -> float:
        """
        Compute the total actual gain produced by a package.

        For every tuple *i* in the package with multiplicity *m_i*:
            contribution_i = (actual_close_at_sell_after_i - stored_price_i) * m_i

        The stored price is the adjusted close as of June 30 of the relation
        year (the same baseline used by the scenario generators).  The actual
        close is looked up from the ActualClose cache; if the exact sell_after
        date is missing, the closest earlier trading day with a price is used.

        Tuples whose ticker has no price data in ActualClose are skipped and
        do not contribute to the returned gain.

        Parameters
        ----------
        package_dict : dict[int, float]
            Mapping  tuple_id  ->  multiplicity.

        Returns
        -------
        float
            Total actual gain in the same currency unit as the stored prices.
        """
        if not package_dict:
            return 0.0

        # 1. Fetch tuple attributes from the relation
        tuple_info = self.__fetch_tuple_info(list(package_dict.keys()))

        # 2. Load actual close series for all distinct tickers in the package
        unique_tickers = list({info[0] for info in tuple_info.values()})
        closes         = self.__fetch_closes(unique_tickers)

        # 3. Accumulate gain
        total_gain    = 0.0
        missing_ticks = set()

        for tid, multiplicity in package_dict.items():
            if tid not in tuple_info:
                continue                           # tuple_id not found in relation

            ticker, sell_after, cutoff_price = tuple_info[tid]

            if sell_after not in self.__sa_to_date:
                continue                           # sell_after out of calendar range

            target_date = self.__sa_to_date[sell_after]

            if ticker not in closes or closes[ticker].empty:
                missing_ticks.add(ticker)
                continue

            series = closes[ticker]
            pos    = series.index.searchsorted(target_date, side='right') - 1
            if pos < 0:
                missing_ticks.add(ticker)
                continue

            actual_price = float(series.iloc[pos])
            total_gain  += (actual_price - cutoff_price) * multiplicity

        if missing_ticks:
            print(
                f'[BackTest] Warning: no ActualClose data for '
                f'{len(missing_ticks)} ticker(s): '
                f'{sorted(missing_ticks)[:10]}'
                + (' …' if len(missing_ticks) > 10 else '')
            )

        return total_gain

    def get_year(self) -> int:
        """Return the calendar year extracted from the relation name."""
        return self.__year

    def get_relation(self) -> str:
        """Return the relation name this BackTest instance is bound to."""
        return self.__relation
