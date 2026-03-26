#!/usr/bin/env python3
"""
Reconcile GBM_Portfolio_YYYY and GARCH_Portfolio_YYYY tables so both contain
exactly the same set of (ticker, sell_after) pairs for each year.

For each year:
1. Compute the intersection of (ticker, sell_after) pairs from both tables.
2. Delete rows in GARCH_Portfolio_YYYY and GBM_Portfolio_YYYY that are NOT
   in the intersection.
3. Remove residuals from GARCH_Residuals_YYYY for tickers no longer present
   in GARCH_Portfolio_YYYY.
4. Renumber IDs in all affected tables sequentially (0-based, ordered by
   original id) to restore contiguous IDs after deletions.

Run from the Data directory or from the project root:
    python Data/reconcile_portfolio_tables.py
"""
import os
from configparser import ConfigParser
import psycopg2

YEARS = list(range(2015, 2026))  # 2015 .. 2025


def _db_config(filename=None, section='postgresql'):
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), 'database.ini')
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


def _renumber_ids(cursor, table: str):
    """
    Renumber the 'id' column of *table* to contiguous integers starting at 0,
    in the order of the current id values.  Works around UNIQUE constraint
    violations by first negating all IDs, then setting final values.
    """
    # Step 1: shift all IDs negative so no conflicts during the final update
    cursor.execute(f'UPDATE {table} SET id = -(id + 1);')

    # Step 2: assign contiguous 0-based IDs ordered by the (negated) id
    cursor.execute(f"""
        WITH ranked AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) - 1 AS new_id
            FROM {table}
        )
        UPDATE {table} t
        SET id = r.new_id
        FROM ranked r
        WHERE t.id = r.id;
    """)


def reconcile_year(cursor, year: int):
    gbm_table    = f'GBM_Portfolio_{year}'
    garch_table  = f'GARCH_Portfolio_{year}'
    resid_table  = f'GARCH_Residuals_{year}'

    # 1. Find intersection of (ticker, sell_after) pairs
    cursor.execute(f"""
        SELECT DISTINCT g.ticker, g.sell_after
        FROM {gbm_table} g
        JOIN {garch_table} c
          ON g.ticker = c.ticker
         AND g.sell_after = c.sell_after;
    """)
    intersection = cursor.fetchall()

    if not intersection:
        print(f'  {year}: intersection is empty — clearing both tables.')
        cursor.execute(f'DELETE FROM {gbm_table};')
        cursor.execute(f'DELETE FROM {garch_table};')
        cursor.execute(f'DELETE FROM {resid_table};')
        return

    # Build temporary table for the intersection
    cursor.execute("""
        CREATE TEMP TABLE _intersection (
            ticker     varchar(10),
            sell_after float
        ) ON COMMIT DROP;
    """)
    cursor.executemany(
        'INSERT INTO _intersection VALUES (%s, %s)',
        intersection,
    )

    # 2. Delete non-intersection rows
    cursor.execute(f"""
        DELETE FROM {gbm_table} t
        WHERE NOT EXISTS (
            SELECT 1 FROM _intersection i
            WHERE i.ticker = t.ticker AND i.sell_after = t.sell_after
        );
    """)
    gbm_deleted = cursor.rowcount

    cursor.execute(f"""
        DELETE FROM {garch_table} t
        WHERE NOT EXISTS (
            SELECT 1 FROM _intersection i
            WHERE i.ticker = t.ticker AND i.sell_after = t.sell_after
        );
    """)
    garch_deleted = cursor.rowcount

    # 3. Remove residuals for tickers that are no longer in GARCH_Portfolio
    cursor.execute(f"""
        DELETE FROM {resid_table} r
        WHERE NOT EXISTS (
            SELECT 1 FROM {garch_table} g WHERE g.ticker = r.ticker
        );
    """)
    resid_deleted = cursor.rowcount

    # Drop temp table before renumbering (it was ON COMMIT DROP, but be explicit)
    cursor.execute('DROP TABLE IF EXISTS _intersection;')

    # 4. Renumber IDs
    _renumber_ids(cursor, gbm_table)
    _renumber_ids(cursor, garch_table)
    _renumber_ids(cursor, resid_table)

    # Row counts after reconciliation
    cursor.execute(f'SELECT COUNT(*) FROM {gbm_table};')
    gbm_count = cursor.fetchone()[0]
    cursor.execute(f'SELECT COUNT(*) FROM {garch_table};')
    garch_count = cursor.fetchone()[0]
    cursor.execute(f'SELECT COUNT(*) FROM {resid_table};')
    resid_count = cursor.fetchone()[0]

    print(f'  {year}: intersection={len(intersection)} pairs  '
          f'deleted GBM={gbm_deleted} GARCH={garch_deleted} residuals={resid_deleted}  '
          f'remaining GBM={gbm_count} GARCH={garch_count} residuals={resid_count}')


def main():
    conn   = _connect()
    cursor = conn.cursor()

    for year in YEARS:
        try:
            reconcile_year(cursor, year)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            print(f'  {year}: ERROR — {exc}')

    cursor.close()
    conn.close()
    print('Reconciliation complete.')


if __name__ == '__main__':
    main()
