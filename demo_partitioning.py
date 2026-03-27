import psycopg2

from DbInfo.GBMPortfolioInfo import GBMPortfolioInfo
from DbInfo.GarchPortfolioInfo import GarchPortfolioInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OfflinePreprocessing.DistPartition import DistPartition
from PgConnection.PgConnection import PgConnection

SIZE_THRESHOLD        = Hyperparameters.SIZE_THRESHOLD
PARTITION_COUNT_LIMIT = int(0.8 * SIZE_THRESHOLD)

YEARS = list(range(2015, 2026))   # 2015 .. 2025


def get_qualifying_relations(prefix):
    """Return [(relation_name, row_count)] for tables above SIZE_THRESHOLD."""
    result = []
    for year in YEARS:
        table = f'{prefix}_{year}'
        try:
            PgConnection.Execute(f'SELECT COUNT(*) FROM {table}')
            count = PgConnection.Fetch()[0][0]
        except psycopg2.errors.UndefinedTable:
            # Table for this year simply does not exist yet — skip silently.
            PgConnection.CONNECTION.rollback()
            continue
        except Exception as e:
            print(f'  Warning: could not query {table}: {e}')
            if PgConnection.CONNECTION is not None:
                PgConnection.CONNECTION.rollback()
            continue
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
    print(f"SIZE_THRESHOLD        = {SIZE_THRESHOLD}")
    print(f"PARTITION_COUNT_LIMIT = {PARTITION_COUNT_LIMIT} (80% of SIZE_THRESHOLD)")
    
    #gbm_relations   = get_qualifying_relations('GBM_Portfolio')
    garch_relations = get_qualifying_relations('GARCH_Portfolio')

    #print(f"\nGBM relations above SIZE_THRESHOLD:   {[r for r, _ in gbm_relations]}")
    print(f"GARCH relations above SIZE_THRESHOLD: {[r for r, _ in garch_relations]}")

    #process_relations(gbm_relations,   GBMPortfolioInfo)
    process_relations(garch_relations, GarchPortfolioInfo)


if __name__ == '__main__':
    main()