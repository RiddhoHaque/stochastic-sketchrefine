# stochastic-sketchrefine

Credentials:

Please have Postgres installed in your environment. Fill up the Postgres database credentials in `Data/database.ini'.

Get a Gurobi WLS license (this is free for academic purposes) from Gurobi Optimization's website.

Add the Gurobi WLS license information in `Utils/GurobiLicense.py'


Database Setup:

Ensure the file 'portfolio.csv' is in the 'Data' folder. Download 'lineitem.tbl' from https://drive.google.com/file/d/1srDPIFyG0LaZOJHeNprSR0UcDHoGQrdf/view?usp=sharing and place it in the Data folder. Then Setup the database by running:

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

Before running stochastic sketchrefine, any relation needs to be partitioned offline. We provide a script that allows users to find out optimal Diameter Thresholds for each dataset, and partition the relation accordingly. Just run:

$ python threshold_search.py

Package Query Processing Runtimes over increasing uncertainties:

To run any package query processing algorithm over increasing uncertainties, use the following command:

$ python main.py inc_variances <dataset> <algName> <queryNumber> <iterations>

where users should replace,

<dataset> by 'tpch' or 'portfolio'
<algName> with 'rclsolve' or 'summarysearch' to run RCLSolve or SummarySearch as the query processing algorithm respectively.
<queryNumber> with any number in [1-5] to run queries 1-5 respectively (Larger numbers can be used if more queries are added to the workload).
<iterations> with any integer to specify the number of iterations.

Example Command:

$ python main.py inc_variances portfolio rclsolve 1 16

You can find the results in the accordingly named files created in Runs/IncVar.

Package Query Processing Runtimes over increasing tuples:

To run any package query processing algorithm over an increasing number of tuples, use the following command:

$ python main.py inc_tuples <dataset> <algName> <queryNumber> <iterations>

where users should replace,

<dataset> by 'tpch' or 'portfolio'
<algName> with 'sskr', 'rclsolve' or 'summarysearch' to run Stochastic SketchRefine, RCLSolve or SummarySearch as the query processing algorithm respectively.
<queryNumber> with any number in [1-5] to run queries 1-5 respectively (Larger numbers can be used if more queries are added to the workload).
<iterations> with any integer to specify the number of iterations.

Example Command:

$ python main.py inc_tuples portfolio sskr 1 16

You can find the results in the accordingly named files created in Runs/IncTuples.

Adding a Custom Dataset:

To add a new dataset, complete the following steps in order.

Step 1 — Implement a ScenarioGenerator for each stochastic attribute.

Create a subclass of ScenarioGenerator (in ScenarioGenerator/) for each stochastic attribute in your dataset. The key method to implement is:

    generate_scenarios(self, seed: int, no_of_scenarios: int) -> list[list[float]]

This method should query the necessary columns from the Postgres relation and return a list with one inner list per tuple, where each inner list contains no_of_scenarios sampled values for that tuple's attribute. Also override generate_scenarios_from_partition, which should join the relation against its partition table (using Relation_Prefixes.PARTITION_RELATION_PREFIX) before calling generate_scenarios — see PriceScenarioGenerator or GainScenarioGenerator for reference implementations.

Step 2 — Create a DbInfo subclass.

Create a subclass of DbInfo (in DbInfo/) and implement the following static methods:

    get_deterministic_attributes()  — return a list of attribute names that are deterministic (i.e. have no scenario generator).
    get_stochastic_attributes()     — return a list of attribute names that are stochastic.
    get_variable_generator_function(attribute)  — return the ScenarioGenerator class (not instance) for the given attribute.
    is_deterministic_attribute(attribute)       — return True if the attribute is in the deterministic list.
    get_diameter_threshold(attribute)           — return the diameter threshold for the given attribute (see Step 3).
    has_inter_tuple_correlations()              — return True if tuples in your dataset have correlated stochastic attributes (e.g. stock prices sharing a ticker), False otherwise.

Step 3 — Set diameter thresholds.

Add a diameter threshold constant to Hyperparameters/Hyperparameters.py for each attribute (both deterministic and stochastic). To find good values automatically, make the following changes and then run threshold_search.py.

In threshold_search.py:

(a) Add a grid dict for your dataset mapping each attribute name to a list of candidate threshold values to search over, following the same format as PORTFOLIO_GRIDS and TPCH_GRIDS at the top of the file.

(b) In get_qualifying_relations(), add your relation names to the tables list.

(c) In main(), add a loop over your relations (following the portfolio and lineitem loops). Inside the loop, construct a DistPartition with your relation and DbInfo subclass, call greedy_threshold_search with your grid dict, and call subprocess.run to invoke OfflineMain.py. The subprocess.run call should pass: your dataset name (as sys.argv[1]), 'distpartition', the relation name, and then each recommended threshold value as a separate argument — one per attribute, in the same order that OfflineMain.py expects them (see step (d)).

In OfflineMain.py:

(d) Add an elif branch for your dataset name. Read the per-attribute threshold values from sys.argv[4] onward (one argument per attribute, matching the order used in step (c)), and assign them to the corresponding Hyperparameters constants. Also set dbInfo to your DbInfo subclass. Check that len(sys.argv) equals 4 plus the number of attributes and exit with an error message if not.

After these changes, run:

$ python threshold_search.py

The recommended threshold values will be printed and the partitioning will be applied automatically. Copy the recommended values into the corresponding constants in Hyperparameters/Hyperparameters.py so that main.py uses them without needing to re-run threshold_search.py.

Step 4 — Write workload queries.

Create a new directory under Workloads/ (e.g. Workloads/MyDataWorkload/) and add one .sql file per query. Queries must follow the stochastic package query syntax accepted by the parser in StochasticPackageQuery/Parser/.

Step 5 — Register the dataset in main.py.

Add an elif branch for your dataset name in main.py (alongside the existing 'tpch' and 'portfolio' branches). Set workload_directory to your new workload folder, dbInfo to your new DbInfo subclass, and populate variance_relations and inc_tuples_relations with the names of the Postgres relations you want to benchmark against.

After completing these steps, pass your dataset name as the <dataset> argument to main.py.