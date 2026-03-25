import sys
import os
import heapq
import subprocess

from DbInfo.PortfolioInfo import PortfolioInfo
from DbInfo.TpchInfo import TpchInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OfflinePreprocessing.DistPartition import DistPartition
from PgConnection.PgConnection import PgConnection

SIZE_THRESHOLD = Hyperparameters.SIZE_THRESHOLD
PARTITION_COUNT_LIMIT = int(0.8 * SIZE_THRESHOLD)

PORTFOLIO_GRIDS = {
    'gain':  [5, 10, 15, 25, 50],
    'price': [1, 5, 10, 15, 50],
}

TPCH_GRIDS = {
    'price':    [5, 10, 30, 50],
    'quantity': [1, 3, 5, 7, 10],
    'tax':      [0.01, 0.02, 0.03],
}


def greedy_threshold_search(dp, grids):
    attrs = list(grids.keys())
    grid_lists = [grids[a] for a in attrs]
    grid_index = [{v: i for i, v in enumerate(g)} for g in grid_lists]

    evaluated = {}

    def get_count(combo):
        if combo not in evaluated:
            thresholds = dict(zip(attrs, combo))
            evaluated[combo] = dp.count_partitions(thresholds)
            print(f"    {thresholds} -> {evaluated[combo]} partitions")
        return evaluated[combo]

    start = tuple(g[0] for g in grid_lists)
    visited = set()
    heap = [(get_count(start), start)]

    while heap:
        _, combo = heapq.heappop(heap)
        if combo in visited:
            continue
        visited.add(combo)

        count = evaluated[combo]
        if count <= PARTITION_COUNT_LIMIT:
            return dict(zip(attrs, combo)), count

        for i in range(len(attrs)):
            idx = grid_index[i][combo[i]]
            if idx + 1 < len(grid_lists[i]):
                neighbor = combo[:i] + (grid_lists[i][idx + 1],) + combo[i + 1:]
                if neighbor not in visited and neighbor not in evaluated:
                    heapq.heappush(heap, (get_count(neighbor), neighbor))

    return None, None


def get_qualifying_relations(prefix):
    tables = [
        'Lineitem_20000',
        'Lineitem_60000',
        'Lineitem_120000',
        'Lineitem_300000',
        'Lineitem_450000',
        'Lineitem_600000',
        'Lineitem_1200000',
        'Lineitem_3000000',
        'Lineitem_4500000',
        'Lineitem_6000000',
        'Stock_Investments_90',
        'Stock_Investments_45',
        'Stock_Investments_30',
        'Stock_Investments_15',
        'Stock_Investments_9',
        'Stock_Investments_3',
        'Stock_Investments_1',
        'Stock_Investments_half',
    ]
    result = []
    for table in tables:
        if not table.lower().startswith(prefix.lower()):
            continue
        PgConnection.Execute(f"SELECT COUNT(*) FROM {table}")
        count = PgConnection.Fetch()[0][0]
        if count > SIZE_THRESHOLD:
            result.append((table, count))
    return result


def main():
    offline_main = os.path.join(os.path.dirname(__file__), 'OfflineMain.py')
    print(f"SIZE_THRESHOLD = {SIZE_THRESHOLD}")
    print(f"PARTITION_COUNT_LIMIT = {PARTITION_COUNT_LIMIT} (60% of SIZE_THRESHOLD)")

    portfolio_relations = get_qualifying_relations('stock_investments')
    lineitem_relations  = get_qualifying_relations('lineitem')

    print(f"\nPortfolio relations above SIZE_THRESHOLD: {[r for r, _ in portfolio_relations]}")
    print(f"Lineitem relations above SIZE_THRESHOLD: {[r for r, _ in lineitem_relations]}")

    for relation, total_tuples in portfolio_relations:
        print(f"\n{'='*60}")
        print(f"PORTFOLIO: {relation} ({total_tuples} tuples)")
        print(f"{'='*60}")
        dp = DistPartition(relation, PortfolioInfo())
        thresholds, count = greedy_threshold_search(dp, PORTFOLIO_GRIDS)
        if thresholds is None:
            print("  No combination found within the partition limit.")
        else:
            print(f"\n  Recommended: {thresholds} -> {count} partitions")
            subprocess.run([
                sys.executable, offline_main,
                'portfolio', 'distpartition', relation,
                str(thresholds['gain']), str(thresholds['price']),
            ], check=True)

    for relation, total_tuples in lineitem_relations:
        print(f"\n{'='*60}")
        print(f"LINEITEM: {relation} ({total_tuples} tuples)")
        print(f"{'='*60}")
        dp = DistPartition(relation, TpchInfo())
        thresholds, count = greedy_threshold_search(dp, TPCH_GRIDS)
        if thresholds is None:
            print("  No combination found within the partition limit.")
        else:
            print(f"\n  Recommended: {thresholds} -> {count} partitions")
            subprocess.run([
                sys.executable, offline_main,
                'tpch', 'distpartition', relation,
                str(thresholds['price']), str(thresholds['quantity']), str(thresholds['tax']),
            ], check=True)


if __name__ == '__main__':
    main()
