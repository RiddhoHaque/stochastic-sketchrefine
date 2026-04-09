import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2

from DbInfo.GBMPortfolioInfo import GBMPortfolioInfo
from DbInfo.GarchPortfolioInfo import GarchPortfolioInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OfflinePreprocessing.DistPartition import DistPartition
from PgConnection.PgConnection import PgConnection

SIZE_THRESHOLD        = Hyperparameters.SIZE_THRESHOLD
PARTITION_COUNT_LIMIT = int(0.8 * SIZE_THRESHOLD)


def get_qualifying_relations(prefix, year):
    """Return [(relation_name, row_count)] for the table matching prefix_year if above SIZE_THRESHOLD."""
    result = []
    table = f'{prefix}_{year}'
    try:
        PgConnection.Execute(f'SELECT COUNT(*) FROM {table}')
        count = PgConnection.Fetch()[0][0]
    except psycopg2.errors.UndefinedTable:
        print(f'  Table {table} does not exist — skipping.')
        PgConnection.CONNECTION.rollback()
        return result
    except Exception as e:
        print(f'  Warning: could not query {table}: {e}')
        if PgConnection.CONNECTION is not None:
            PgConnection.CONNECTION.rollback()
        return result
    if count > SIZE_THRESHOLD:
        result.append((table, count))
    return result


def process_relations(relations, db_info_class):
    for relation, _ in relations:
        partitioner = DistPartition(relation=relation, dbInfo=db_info_class())
        partitioner.partition_relation()
        print(f'  {relation}: {partitioner.get_no_of_partitions():,} partitions')
        metrics = partitioner.get_metrics()
        metrics.log_performance()


def main():
    if len(sys.argv) != 2:
        print(f'Usage: python {sys.argv[0]} <year>')
        sys.exit(1)
    try:
        year = int(sys.argv[1])
    except ValueError:
        print(f'Error: year must be an integer, got {sys.argv[1]!r}')
        sys.exit(1)

    print(f"Year                  = {year}")
    print(f"SIZE_THRESHOLD        = {SIZE_THRESHOLD}")
    print(f"PARTITION_COUNT_LIMIT = {PARTITION_COUNT_LIMIT} (80% of SIZE_THRESHOLD)")

    gbm_relations   = get_qualifying_relations('GBM_Portfolio',   year)
    garch_relations = get_qualifying_relations('GARCH_Portfolio', year)

    print(f"\nGBM relations above SIZE_THRESHOLD:   {[r for r, _ in gbm_relations]}")
    print(f"GARCH relations above SIZE_THRESHOLD: {[r for r, _ in garch_relations]}")

    process_relations(gbm_relations,   GBMPortfolioInfo)
    process_relations(garch_relations, GarchPortfolioInfo)


if __name__ == '__main__':
    main()
