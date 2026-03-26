#!/usr/bin/env bash
# Data/setup_demo_data.sh
# Build all portfolio tables, populate them from Yahoo Finance, then reconcile.
# Run from the Data directory OR from the project root.

set -e   # exit immediately on any error

# Resolve the Data directory (where this script lives) and the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Step 1 & 2: create table schemas (table builders use relative paths, so
#    run them from inside Data/).
cd "$SCRIPT_DIR"

echo "=== Step 1: Create GARCH table schemas ==="
python garch_table_builder.py
echo "GARCH schemas created."

echo ""
echo "=== Step 2: Create GBM table schemas ==="
python gbm_table_builder.py
echo "GBM schemas created."

# ── Steps 3-5: fetchers and reconcile use os.path.dirname(__file__) for
#    database.ini, so they work from any directory.  Run from the project root
#    so that any project-level imports (PgConnection, etc.) resolve correctly.
cd "$PROJECT_ROOT"

echo ""
echo "=== Step 3: Populate GARCH tables from Yahoo Finance ==="
python Data/Portfolio/yfinance_garch_fetcher.py

echo ""
echo "=== Step 4: Populate GBM tables from Yahoo Finance ==="
python Data/Portfolio/yfinance_gbm_fetcher.py

echo ""
echo "=== Step 5: Reconcile GARCH and GBM tables ==="
python Data/reconcile_portfolio_tables.py

echo ""
echo "=== All done! ==="
