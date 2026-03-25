import sys
from DbInfo.DbInfo import DbInfo
from DbInfo.PortfolioInfo import PortfolioInfo
from DbInfo.TpchInfo import TpchInfo
from Hyperparameters.Hyperparameters import Hyperparameters
from OfflinePreprocessing.DistPartition import DistPartition


dbInfo = DbInfo

if str(sys.argv[1]) == 'tpch':
    dbInfo = TpchInfo
    if len(sys.argv) != 7:
        print('No of arguments incorrect')
        sys.exit(0)
    Hyperparameters.DIAMETER_THRESHOLD_TPCH_PRICE = float(sys.argv[4])
    Hyperparameters.DIAMETER_THRESHOLD_TPCH_QUANTITY = float(sys.argv[5])
    Hyperparameters.DIAMETER_THRESHOLD_TPCH_TAX = float(sys.argv[6])
        

elif str(sys.argv[1]) == 'portfolio':
    dbInfo = PortfolioInfo
    if len(sys.argv) != 6:
        print('No of arguments incorrect')
        sys.exit(0)
    Hyperparameters.DIAMETER_THRESHOLD_PORTFOLIO_GAIN = float(sys.argv[4])
    Hyperparameters.DIAMETER_THRESHOLD_PORTFOLIO_PRICE = float(sys.argv[5])
        
else:
    print('Database not specified correctly.')
    sys.exit(0)

partition_class = None
if str(sys.argv[2]) == 'distpartition':
    partition_class = DistPartition
else:
    print('Partitioning algorithm not specified correctly.')
    sys.exit(0)

relation_name = str(sys.argv[3])

partitioner = partition_class(relation=relation_name, dbInfo=dbInfo)
partitioner.partition_relation()
partitioner_metrics = partitioner.get_metrics()
partitioner_metrics.log_performance()

