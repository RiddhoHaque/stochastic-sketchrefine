#!/usr/bin/env python3
"""
Benchmark GBM vs GARCH-FHS gain scenario generators.

Evaluates prediction errors at sell_after = 10, 100, and 200 trading days
for the year-pairs 2015-16, 2017-18, 2019-20, and 2021-22.

Actual closing prices are downloaded from Yahoo Finance on the first run and
stored in PostgreSQL (ActualClose / ActualCloseDownloaded tables), so every
subsequent run skips the network download entirely.

Run from project root:
    python benchmark_gain_methods.py

Output:
    Evaluation/benchmark_results.json  -- per-SA errors and runtimes
    Console: mean runtimes for each method
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from configparser import ConfigParser
import psycopg2
import psycopg2.extras

from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.PorfolioScenarioGenerator.GBMGainScenarioGenerator import (
    GBMGainScenarioGenerator,
)
from ScenarioGenerator.PorfolioScenarioGenerator.GARCHFHSGainScenarioGenerator import (
    GARCHFHSGainScenarioGenerator,
)

# July-1 cutoff years to evaluate (each covers Jul Y – Jun Y+1)
YEARS = [2015, 2017, 2019, 2021]
# Sell-after checkpoints (in NYSE trading days) to evaluate
TARGET_SELL_AFTERS = [10, 100, 200]

NO_OF_SCENARIOS = 100
SEED = 42
OUTPUT_DIR = 'Evaluation'
DOWNLOAD_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# DB connection helpers
# ---------------------------------------------------------------------------

def _db_config():
    filename = os.path.join('Data', 'database.ini')
    parser = ConfigParser()
    parser.read(filename)
    cfg = {}
    if 'postgresql' in parser:
        for key in parser['postgresql']:
            cfg[key] = parser['postgresql'][key]
    return cfg


def _connect():
    cfg = _db_config()
    return psycopg2.connect(
        dbname=cfg['dbname'], user=cfg['user'],
        host=cfg['host'], password=cfg['password'], port=cfg['port'],
    )


def query_param_rows(conn, table):
    """Return ordered list of (ticker, sell_after, price) for all rows."""
    sql = f'SELECT ticker, sell_after, price FROM {table} ORDER BY id;'
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    return rows


# ---------------------------------------------------------------------------
# Actual-close price cache (PostgreSQL)
# ---------------------------------------------------------------------------

def ensure_cache_tables(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ActualClose (
            ticker  varchar(10)      NOT NULL,
            date    date             NOT NULL,
            close   double precision NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ActualCloseDownloaded (
            ticker      varchar(10) NOT NULL,
            period_year int         NOT NULL,
            PRIMARY KEY (ticker, period_year)
        )
    """)
    conn.commit()
    cur.close()


def get_cached_tickers(conn, tickers, year):
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker FROM ActualCloseDownloaded "
        "WHERE period_year = %s AND ticker = ANY(%s)",
        (year, list(tickers)),
    )
    rows = cur.fetchall()
    cur.close()
    return {r[0] for r in rows}


def store_closes(conn, year, ticker_closes, no_data_tickers=None):
    cur = conn.cursor()

    price_rows = []
    for ticker, series in ticker_closes.items():
        for ts, price in series.items():
            if pd.isna(price) or price <= 0:
                continue
            d = ts.date() if hasattr(ts, 'date') else ts
            price_rows.append((ticker, d, float(price)))

    if price_rows:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO ActualClose (ticker, date, close) VALUES %s "
            "ON CONFLICT (ticker, date) DO NOTHING",
            price_rows,
            page_size=2000,
        )

    all_tickers = list(ticker_closes.keys())
    if no_data_tickers:
        all_tickers += list(no_data_tickers)
    if all_tickers:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO ActualCloseDownloaded (ticker, period_year) VALUES %s "
            "ON CONFLICT DO NOTHING",
            [(t, year) for t in all_tickers],
        )

    conn.commit()
    cur.close()


def query_cached_closes(conn, tickers, start_date, end_date):
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, date, close FROM ActualClose "
        "WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s "
        "ORDER BY ticker, date",
        (list(tickers), start_date.date(), end_date.date()),
    )
    rows = cur.fetchall()
    cur.close()

    data: dict = {}
    for ticker, date, close in rows:
        data.setdefault(ticker, {})[pd.Timestamp(date)] = close
    return {t: pd.Series(v) for t, v in data.items()}


# ---------------------------------------------------------------------------
# yfinance download
# ---------------------------------------------------------------------------

def _normalise_index(obj):
    obj = obj.copy()
    idx = obj.index.normalize()
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    obj.index = idx
    return obj


def _download_from_yfinance(tickers, start_date, end_date):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str   = (end_date + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    result = {}
    unique = list(dict.fromkeys(tickers))

    for batch_start in range(0, len(unique), DOWNLOAD_BATCH_SIZE):
        batch = unique[batch_start: batch_start + DOWNLOAD_BATCH_SIZE]
        try:
            raw = yf.download(
                batch, start=start_str, end=end_str,
                progress=False, auto_adjust=True, actions=False,
            )
        except Exception as exc:
            print(f'    yfinance batch error: {exc}')
            continue
        if raw.empty:
            continue
        close = raw['Close']
        if isinstance(close, pd.Series):
            close = close.to_frame(batch[0])
        close = _normalise_index(close)
        for ticker in batch:
            if ticker in close.columns:
                s = close[ticker].dropna()
                if not s.empty:
                    result[ticker] = s

    return result


def load_or_download_closes(conn, tickers, year, start_date, end_date):
    tickers_set = set(tickers)
    already     = get_cached_tickers(conn, tickers_set, year)
    missing     = tickers_set - already

    if missing:
        print(f'    {len(already)} tickers in cache; downloading {len(missing)} from yfinance …')
        downloaded = _download_from_yfinance(list(missing), start_date, end_date)
        no_data    = missing - set(downloaded.keys())
        store_closes(conn, year, downloaded, no_data_tickers=no_data)
        if no_data:
            print(f'    Stored {len(downloaded)} tickers ({len(no_data)} had no yfinance data).')
        else:
            print(f'    Stored {len(downloaded)} tickers in ActualClose cache.')
    else:
        print(f'    All {len(tickers_set)} tickers already in cache — skipping download.')

    return query_cached_closes(conn, list(tickers_set), start_date, end_date)


# ---------------------------------------------------------------------------
# Error calculation
# ---------------------------------------------------------------------------

def compute_errors_per_sa(param_rows, scenarios, sa_to_date, actual_close,
                           target_sas):
    """
    For each sell_after in target_sas compute per-ticker absolute prediction error:
        abs( mean_simulated_gain[sa] - actual_gain[sa] )

    Returns dict  sa_int -> {ticker: abs_error}.
    """
    # Build ticker -> {sa: [scenario gains], price}  (only target sell_afters)
    target_set = set(target_sas)
    ticker_data: dict = {}
    for idx, (ticker, sell_after, price) in enumerate(param_rows):
        sa = int(sell_after)
        if sa not in target_set:
            continue
        if ticker not in ticker_data:
            ticker_data[ticker] = {'price': float(price), 'sa_gains': {}}
        if idx < len(scenarios) and len(scenarios[idx]) > 0:
            ticker_data[ticker]['sa_gains'][sa] = scenarios[idx]

    errors_by_sa: dict = {sa: {} for sa in target_sas}

    for ticker, tdata in ticker_data.items():
        if ticker not in actual_close:
            continue
        closes       = actual_close[ticker]
        cutoff_price = tdata['price']

        for sa, sim_gains in tdata['sa_gains'].items():
            if sa not in sa_to_date:
                continue
            date    = sa_to_date[sa]
            idx_pos = closes.index.searchsorted(date, side='right') - 1
            if idx_pos < 0:
                continue
            actual_price = float(closes.iloc[idx_pos])
            if np.isnan(actual_price) or actual_price <= 0:
                continue
            actual_gain = actual_price - cutoff_price
            mean_sim    = float(np.mean(sim_gains))
            errors_by_sa[sa][ticker] = abs(mean_sim - actual_gain)

    return errors_by_sa


# ---------------------------------------------------------------------------
# NYSE calendar helper
# ---------------------------------------------------------------------------

def get_sell_after_dates(year):
    nyse  = mcal.get_calendar('NYSE')
    start = pd.Timestamp(year=year, month=7, day=1)
    end   = pd.Timestamp(year=year + 1, month=6, day=30)
    sched = nyse.schedule(start_date=start, end_date=end)
    dates = sched.index.normalize()
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    return {i + 1: dates[i] for i in range(len(dates))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = _connect()
    gbm_runtimes:        list = []
    garch_runtimes:      list = []
    gbm_runtime_years:   list = []
    garch_runtime_years: list = []
    all_gbm_errors:   dict = {}   # year -> {sa -> {ticker -> abs_error}}
    all_garch_errors: dict = {}

    try:
        ensure_cache_tables(conn)

        for year in YEARS:
            gbm_table   = f'GBM_Portfolio_{year}'
            garch_table = f'GARCH_Portfolio_{year}'
            print(f'\n=== Year {year}-{year+1} ===')

            try:
                gbm_rows   = query_param_rows(conn, gbm_table)
                garch_rows = query_param_rows(conn, garch_table)
            except Exception as exc:
                print(f'  DB query error: {exc}; skipping year.')
                conn.rollback()
                continue

            if not gbm_rows and not garch_rows:
                print('  No data, skipping.')
                continue
            print(f'  GBM rows: {len(gbm_rows):,}  |  GARCH param rows: {len(garch_rows):,}')

            # ---- generate scenarios with timing ----
            gbm_scenarios = []
            if gbm_rows:
                gbm_gen = GBMGainScenarioGenerator(gbm_table)
                t0 = time.perf_counter()
                gbm_scenarios = gbm_gen.generate_scenarios(SEED, NO_OF_SCENARIOS)
                gbm_time = time.perf_counter() - t0
                gbm_runtimes.append(gbm_time)
                gbm_runtime_years.append(year)
                print(f'  GBM       generated in {gbm_time:.2f}s')

            garch_scenarios = []
            if garch_rows:
                garch_gen = GARCHFHSGainScenarioGenerator(garch_table)
                t0 = time.perf_counter()
                garch_scenarios = garch_gen.generate_scenarios(SEED, NO_OF_SCENARIOS)
                garch_time = time.perf_counter() - t0
                garch_runtimes.append(garch_time)
                garch_runtime_years.append(year)
                print(f'  GARCH-FHS generated in {garch_time:.2f}s')

            # ---- sell_after -> calendar date mapping ----
            sa_to_date = get_sell_after_dates(year)

            # ---- load/cache actual closes ----
            all_tickers = list(dict.fromkeys(
                [r[0] for r in gbm_rows] + [r[0] for r in garch_rows]
            ))
            start_date = pd.Timestamp(year=year,     month=7, day=1)
            end_date   = pd.Timestamp(year=year + 1, month=6, day=30)

            print(f'  Checking price cache for {len(all_tickers)} tickers …')
            actual_close = load_or_download_closes(conn, all_tickers, year,
                                                   start_date, end_date)
            print(f'  Price data available for {len(actual_close)} tickers.')

            # ---- compute per-SA per-ticker absolute errors ----
            gbm_errors   = compute_errors_per_sa(gbm_rows,   gbm_scenarios,
                                                 sa_to_date, actual_close,
                                                 TARGET_SELL_AFTERS)
            garch_errors = compute_errors_per_sa(garch_rows, garch_scenarios,
                                                 sa_to_date, actual_close,
                                                 TARGET_SELL_AFTERS)
            all_gbm_errors[year]   = gbm_errors
            all_garch_errors[year] = garch_errors

            for sa in TARGET_SELL_AFTERS:
                n_gbm   = len(gbm_errors.get(sa, {}))
                n_garch = len(garch_errors.get(sa, {}))
                print(f'  SA={sa:3d}: GBM={n_gbm} tickers, GARCH-FHS={n_garch} tickers')

    finally:
        conn.close()

    # ---- runtime summary ----
    print('\n' + '=' * 55)
    print('Runtime Summary  (100 scenarios per call)')
    print('=' * 55)
    if gbm_runtimes:
        print(f'  GBM        mean={np.mean(gbm_runtimes):.2f}s  '
              f'median={np.median(gbm_runtimes):.2f}s  '
              f'min={np.min(gbm_runtimes):.2f}s  '
              f'max={np.max(gbm_runtimes):.2f}s')
    else:
        print('  GBM        no data')
    if garch_runtimes:
        print(f'  GARCH-FHS  mean={np.mean(garch_runtimes):.2f}s  '
              f'median={np.median(garch_runtimes):.2f}s  '
              f'min={np.min(garch_runtimes):.2f}s  '
              f'max={np.max(garch_runtimes):.2f}s')
    else:
        print('  GARCH-FHS  no data')
    print('=' * 55)

    # ---- write results JSON ----
    gbm_year_to_time   = dict(zip(gbm_runtime_years,   gbm_runtimes))
    garch_year_to_time = dict(zip(garch_runtime_years, garch_runtimes))

    results = {
        'no_of_scenarios':    NO_OF_SCENARIOS,
        'years':              YEARS,
        'target_sell_afters': TARGET_SELL_AFTERS,
        # gbm_errors[year][sa][ticker] = abs_error
        'gbm_errors': {
            str(y): {str(sa): errs
                     for sa, errs in sa_dict.items()}
            for y, sa_dict in all_gbm_errors.items()
        },
        'garch_errors': {
            str(y): {str(sa): errs
                     for sa, errs in sa_dict.items()}
            for y, sa_dict in all_garch_errors.items()
        },
        'gbm_runtimes':   {str(y): t for y, t in gbm_year_to_time.items()},
        'garch_runtimes': {str(y): t for y, t in garch_year_to_time.items()},
    }

    json_path = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults written to {json_path}')
    print('Run plot_benchmark_results.py to generate plots.')


if __name__ == '__main__':
    main()
