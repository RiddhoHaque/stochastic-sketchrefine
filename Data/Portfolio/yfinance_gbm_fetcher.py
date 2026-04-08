import os
import math
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from configparser import ConfigParser
import psycopg2
import psycopg2.extras

CUTOFF_YEARS = list(range(2015, 2026))  # 2015 .. 2025
CUTOFF_MONTH = 6
CUTOFF_DAY = 30
MIN_RETURNS = 252       # require at least 1 year of daily returns
HISTORY_START = '2010-01-01'

_STOCK_DATA_DIRS = [
    'stock_market_data/nasdaq/csv',
    'stock_market_data/nyse/csv',
]


def _db_config(filename=None, section='postgresql'):
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), '..', 'database.ini')
    parser = ConfigParser()
    parser.read(filename)
    cfg = {}
    if section in parser:
        for key in parser[section]:
            cfg[key] = parser[section][key]
    return cfg


def _connect():
    cfg = _db_config()
    return psycopg2.connect(
        dbname=cfg['dbname'],
        user=cfg['user'],
        host=cfg['host'],
        password=cfg['password'],
        port=cfg['port'],
    )


def _fetch_tickers() -> list:
    """Collect tickers from local NASDAQ and NYSE stock_market_data CSV directories."""
    base = os.path.dirname(__file__)
    tickers = set()
    for rel_dir in _STOCK_DATA_DIRS:
        directory = os.path.join(base, rel_dir)
        if not os.path.isdir(directory):
            print(f'  Warning: directory not found: {directory}')
            continue
        for fname in os.listdir(directory):
            if fname.endswith('.csv'):
                tickers.add(fname[:-4])
    return sorted(tickers)


def _build_sa_to_date_by_year(years: list) -> dict:
    """Pre-compute {year: {sell_after_int: pd.Timestamp}} for all years."""
    nyse = mcal.get_calendar('NYSE')
    result = {}
    for year in years:
        start = pd.Timestamp(year=year, month=7, day=1)
        end   = pd.Timestamp(year=year + 1, month=6, day=30)
        sched = nyse.schedule(start_date=start, end_date=end)
        dates = sched.index.normalize()
        if dates.tz is not None:
            dates = dates.tz_convert(None)
        result[year] = {i + 1: dates[i] for i in range(len(dates))}
    return result


def _align_to_trading_calendar(closes: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    """Reindex closes to NYSE trading days up to cutoff and forward-fill missing days."""
    nyse = mcal.get_calendar('NYSE')
    idx = closes.index.normalize()
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    closes = closes.copy()
    closes.index = idx
    schedule = nyse.schedule(start_date=closes.index[0], end_date=cutoff)
    trading_days = schedule.index
    if trading_days.tz is not None:
        trading_days = trading_days.tz_convert(None)
    return closes.reindex(trading_days).ffill().dropna()


def _fit_gbm(log_returns: np.ndarray):
    """Estimate GBM parameters from daily log returns."""
    alpha      = float(np.mean(log_returns))
    volatility = float(np.std(log_returns))
    drift      = alpha + 0.5 * volatility ** 2
    return volatility, drift


# ---------------------------------------------------------------------------
# Raw-close cache helpers
# ---------------------------------------------------------------------------

def _ensure_raw_close_tables(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS RawClose (
            ticker varchar(10)      NOT NULL,
            date   date             NOT NULL,
            close  double precision NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS RawCloseDownloaded (
            ticker varchar(10) NOT NULL PRIMARY KEY
        )
    """)


def _is_ticker_downloaded(cursor, ticker: str) -> bool:
    cursor.execute("SELECT 1 FROM RawCloseDownloaded WHERE ticker=%s", (ticker,))
    return cursor.fetchone() is not None


def _load_closes_from_cache(cursor, ticker: str) -> pd.Series:
    cursor.execute(
        "SELECT date, close FROM RawClose WHERE ticker=%s ORDER BY date",
        (ticker,),
    )
    rows = cursor.fetchall()
    if not rows:
        return pd.Series(dtype=float)
    dates, closes = zip(*rows)
    return pd.Series(list(closes), index=pd.to_datetime(dates))


def _cache_closes(cursor, ticker: str, closes_full: pd.Series):
    rows = [(ticker, d.date(), float(v)) for d, v in closes_full.items()]
    if rows:
        psycopg2.extras.execute_values(
            cursor,
            "INSERT INTO RawClose (ticker, date, close) VALUES %s "
            "ON CONFLICT (ticker, date) DO NOTHING",
            rows,
        )
    cursor.execute(
        "INSERT INTO RawCloseDownloaded (ticker) VALUES (%s) ON CONFLICT DO NOTHING",
        (ticker,),
    )


def _remove_ticker_from_directories(ticker: str):
    base = os.path.dirname(__file__)
    for rel_dir in _STOCK_DATA_DIRS:
        path = os.path.join(base, rel_dir, f'{ticker}.csv')
        if os.path.isfile(path):
            os.remove(path)
            print(f'  Removed {path}')


# ---------------------------------------------------------------------------
# ActualClose helpers
# ---------------------------------------------------------------------------

def _ensure_actual_close_tables(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ActualClose (
            ticker  varchar(10)      NOT NULL,
            date    date             NOT NULL,
            close   double precision NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ActualCloseDownloaded (
            ticker      varchar(10) NOT NULL,
            period_year int         NOT NULL,
            PRIMARY KEY (ticker, period_year)
        )
    """)


def _is_year_cached(cursor, ticker: str, year: int) -> bool:
    cursor.execute(
        "SELECT 1 FROM ActualCloseDownloaded WHERE ticker=%s AND period_year=%s",
        (ticker, year),
    )
    return cursor.fetchone() is not None


def _get_valid_sa_from_cache(cursor, ticker: str, sa_to_date: dict) -> list:
    """Return sell_after ints whose exact sell_after date is present in ActualClose."""
    dates = [d.date() for d in sa_to_date.values()]
    cursor.execute(
        "SELECT date FROM ActualClose WHERE ticker=%s AND date=ANY(%s)",
        (ticker, dates),
    )
    cached = {r[0] for r in cursor.fetchall()}
    return sorted(sa for sa, dt in sa_to_date.items() if dt.date() in cached)


def _store_sa_prices(cursor, ticker: str, year: int,
                     closes_full: pd.Series, sa_to_date: dict) -> list:
    """
    Extract sell_after prices from closes_full (exact date match only — no forward-fill),
    store in ActualClose, mark year as downloaded, and return valid sell_after ints.
    """
    rows = []
    valid_sa = []
    for sa, date in sa_to_date.items():
        if date in closes_full.index:
            price = float(closes_full.loc[date])
            if not (math.isnan(price) or price <= 0):
                rows.append((ticker, date.date(), price))
                valid_sa.append(sa)
    if rows:
        psycopg2.extras.execute_values(
            cursor,
            "INSERT INTO ActualClose (ticker, date, close) VALUES %s "
            "ON CONFLICT (ticker, date) DO NOTHING",
            rows,
        )
    cursor.execute(
        "INSERT INTO ActualCloseDownloaded (ticker, period_year) VALUES (%s, %s) "
        "ON CONFLICT DO NOTHING",
        (ticker, year),
    )
    return sorted(valid_sa)


# ---------------------------------------------------------------------------
# Insert helper
# ---------------------------------------------------------------------------

def _insert_ticker(cursor, year: int, row_ids: dict, ticker: str,
                   price: float, volatility: float, drift: float,
                   valid_sell_afters: list):
    table  = f'GBM_Portfolio_{year}'
    row_id = row_ids[year]
    for sell_after in valid_sell_afters:
        cursor.execute(
            f'INSERT INTO {table} VALUES (%s,%s,%s,%s,%s,%s);',
            (row_id, ticker, float(sell_after), price, volatility, drift),
        )
        row_id += 1
    row_ids[year] = row_id


def main():
    tickers = _fetch_tickers()
    print(f'Fetched {len(tickers)} tickers from local NASDAQ/NYSE directories.')

    sa_to_date_by_year = _build_sa_to_date_by_year(CUTOFF_YEARS)
    cutoffs = {
        year: pd.Timestamp(year=year, month=CUTOFF_MONTH, day=CUTOFF_DAY)
        for year in CUTOFF_YEARS
    }

    # Extend download window to cover all sell_after dates
    last_year    = CUTOFF_YEARS[-1]
    last_sa_end  = pd.Timestamp(year=last_year + 1, month=6, day=30)
    today        = pd.Timestamp.today().normalize()
    if today.tz is not None:
        today = today.tz_convert(None)
    global_end     = min(last_sa_end, today) + pd.Timedelta(days=1)
    global_end_str = global_end.strftime('%Y-%m-%d')

    conn   = _connect()
    cursor = conn.cursor()
    _ensure_actual_close_tables(cursor)
    _ensure_raw_close_tables(cursor)
    conn.commit()

    row_ids = {year: 0 for year in CUTOFF_YEARS}

    for i, ticker in enumerate(tickers):
        try:
            if _is_ticker_downloaded(cursor, ticker):
                closes_full = _load_closes_from_cache(cursor, ticker)
                if closes_full.empty:
                    continue
            else:
                try:
                    raw = yf.download(
                        ticker,
                        start=HISTORY_START,
                        end=global_end_str,
                        progress=False,
                        auto_adjust=True,
                        actions=False,
                    )
                except Exception as e:
                    print(f'  Download failed for {ticker}: {e}')
                    _remove_ticker_from_directories(ticker)
                    continue

                if raw.empty:
                    _remove_ticker_from_directories(ticker)
                    continue

                closes_full = raw['Close'].squeeze().dropna()
                if closes_full.index.tz is not None:
                    closes_full.index = closes_full.index.tz_convert(None)
                closes_full.index = closes_full.index.normalize()

                _cache_closes(cursor, ticker, closes_full)
                conn.commit()

            for year in CUTOFF_YEARS:
                saved_id = row_ids[year]
                try:
                    cursor.execute('SAVEPOINT sp_year')

                    # Determine valid sell_after values
                    sa_to_date = sa_to_date_by_year[year]
                    if _is_year_cached(cursor, ticker, year):
                        valid_sa = _get_valid_sa_from_cache(cursor, ticker, sa_to_date)
                    else:
                        valid_sa = _store_sa_prices(cursor, ticker, year,
                                                    closes_full, sa_to_date)

                    if not valid_sa:
                        cursor.execute('RELEASE SAVEPOINT sp_year')
                        continue

                    # Fit GBM on historical data up to cutoff
                    cutoff  = cutoffs[year]
                    slice_  = closes_full.loc[:cutoff]
                    if slice_.empty:
                        cursor.execute('RELEASE SAVEPOINT sp_year')
                        continue
                    closes  = _align_to_trading_calendar(slice_, cutoff)
                    if len(closes) < MIN_RETURNS + 1:
                        cursor.execute('RELEASE SAVEPOINT sp_year')
                        continue

                    log_returns = np.log(closes / closes.shift(1)).dropna().values.astype(float)
                    price = float(closes.iloc[-1])
                    if math.isnan(price) or price <= 0:
                        cursor.execute('RELEASE SAVEPOINT sp_year')
                        continue

                    volatility, drift = _fit_gbm(log_returns)

                    if any(math.isnan(v) or math.isinf(v) for v in [volatility, drift]):
                        cursor.execute('RELEASE SAVEPOINT sp_year')
                        continue
                    if volatility <= 0:
                        cursor.execute('RELEASE SAVEPOINT sp_year')
                        continue

                    _insert_ticker(cursor, year, row_ids, ticker,
                                   price, volatility, drift, valid_sa)
                    cursor.execute('RELEASE SAVEPOINT sp_year')

                except Exception as e:
                    cursor.execute('ROLLBACK TO SAVEPOINT sp_year')
                    row_ids[year] = saved_id
                    print(f'  Skipping {ticker} / {year}: {e}')
                    continue

        except Exception as e:
            print(f'  Skipping {ticker}: {e}')
            continue

        if (i + 1) % 100 == 0:
            conn.commit()
            print(f'  {i + 1}/{len(tickers)} tickers processed.')

    conn.commit()
    for year in CUTOFF_YEARS:
        print(f'  GBM_Portfolio_{year}: {row_ids[year]} rows inserted.')
    cursor.close()
    conn.close()
    print('Done.')


if __name__ == '__main__':
    main()
