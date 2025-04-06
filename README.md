# stochastic-sketchrefine

Credentials:

Please have Postgres installed in your environment. Fill up the Postgres database credentials in `Data/database.ini'.

Get a Gurobi WLS license (this is free for academic purposes) from Gurobi Optimization's website.

Add the Gurobi WLS license information in `Utils/GurobiLicense.py'


Database Setup:

Ensure the file 'portfolio.csv' is in the 'Data' folder. The TPC-H dbgen script can be used to build an instance of the lineitem table. Please refer to the readme in the TPC-H directory inside the Data folder for more information. Rename the generated file as 'lineitem.tbl' and place it in the Data folder. Then Setup the database by running:

$ ./dataset_creator.sh

Version Stability Check:

This repository may be updated regularly to implement future tweaks to our algorithms, fix bugs and run more experiments. Some of these changes may result in unwanted software behaviour. As an initial check to see if the current version is stable, users are encourages to run:

$ python UnitTestRunner.py

In the unlikely event that any of the unit tests fail, the final repository may have a bug, and the final commit(s) could be reverted until a stable version is found where all the tests pass, as will be confirmed by the 'ALL TESTS PASSED' log that appears in the end. Note that passing all unit tests does not necessarily ensure the state of the codebase is stable and bug-free. In case you face any other issues or, please contact us via rhaque@umass.edu.

Workload Preparation:

We provide 5 example queries for both the TPC-H and Portfolio datasets in the 'Workloads/PortfolioWorkload' and 'Workloads/TpchWorkload' directories. Users can update these queries or add new syntactically correct ones in the same directories.

Updating Hyperparameters:

We provide all default hyperparameter settings in 'Hyperparameters/Hyperparameters.py'. Users can tweak their values by updating the file directly.

Offline pre-processing:

Before running stochastic sketchrefine, any relation needs to be partitioned offline. We implemented four different partitioning schemes, that are later combined with our proposed algorithms for representative selection and - in case of inter-tuple correlations - computes histograms of each representative's PDFs, and determines required correlations between representative's duplicates. For partitioning all relations of a database offline, use the command:

$ python OfflinePreprocessing/DistPartition.py <Dataset> <PartitioningAlg>

where users should replace:

<Dataset> by 'tpch' or 'portfolio'
<PartitioningAlg> by 'distpartition', 'kdtrees', 'pca' or 'findit' to use DistPartition, KD-Trees, PCA + Agglomerative Clustering, or FINDIT respectively to partition the relations.

Example Command:

$ python OfflinePreprocessing/DistPartition.py portfolio distpartition

The logs will show the runtimes needed for partitioning and the entire offline preprocessing phase for e.

Package Query Processing Runtimes over increasing uncertainties:

To run any package query processing algorithm over increasing uncertainties, use the following command:

$ python main.py inc_variances <dataset> <algName> <queryNumber> <iterations>

where users should replace,

<dataset> by 'tpch' or 'portfolio'
<algName> with 'rclsolve' or 'summarysearch' to run RCLSolve or SummarySearch as the query processing algorithm respectively.
<queryNumber> with any number in [1-5] to run queries 1-5 respectively (Larger numbers can be used if more queries are added to the workload).
<iterations> with any integer to specify the number of iterations.

Example Command:

$ python main.py inc_variances portfolio sskr 1 16

The resulting logs will indicate runtimes, means and standard deviations of relative integrality gaps, the generated packages and, for all packages, if all probabilistic constraints are satisfied among 10^6 newly generated test scenarios (with successful verifications indicated by the log 'Package satisfies probabilistic constraints over test scenarios with an objective value of x', x being the objective value over the test scenarios).

Package Query Processing Runtimes over increasing tuples:

To run any package query processing algorithm over an increasing number of tuples, use the following command:

$ python main.py inc_tuples <dataset> <minTupleCount> <maxTupleCount> <algName> <queryNumber> <iterations>

where users should replace,

<dataset> by 'tpch' or 'portfolio'
<minTupleCount> with an integer to set the size of the lowest number of tuples for which queries will be executed. For example, setting maxTupleCount to 100 will run the algorithm for all relations in the dataset with 100 tuples or more.
<maxTupleCount> with an integer to set the size of the highest number of tuples for which queries will be executed. For example, setting maxTupleCount to 100,000 will run the algorithm for all relations in the dataset with 100,000 tuples or less.
<algName> with 'sskr', 'rclsolve' or 'summarysearch' to run Stochastic SketchRefine, RCLSolve or SummarySearch as the query processing algorithm respectively.
<queryNumber> with any number in [1-5] to run queries 1-5 respectively (Larger numbers can be used if more queries are added to the workload).
<iterations> with any integer to specify the number of iterations.

Example Command:

$ python main.py inc_tuples portfolio 1 10000000 sskr 1 16

The resulting logs will indicate runtimes, means and standard deviations of relative integrality gaps, the generated packages and, for all packages, if all probabilistic constraints are satisfied among 10^6 newly generated test scenarios (with successful verifications indicated by the log 'Package satisfies probabilistic constraints over test scenarios with an objective value of x', x being the objective value over the test scenarios).