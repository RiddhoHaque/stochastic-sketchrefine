@echo off
:: Data\setup_demo_data.bat
:: Build all portfolio tables, populate them from Yahoo Finance, then reconcile.
:: Run from the Data directory OR from the project root.

:: Resolve the Data directory (where this script lives) and the project root.
set SCRIPT_DIR=%~dp0
:: Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
for %%i in ("%SCRIPT_DIR%") do set PROJECT_ROOT=%%~dpi
if "%PROJECT_ROOT:~-1%"=="\" set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

echo === Step 1: Create GARCH table schemas ===
cd /d "%SCRIPT_DIR%"
python garch_table_builder.py || goto :error
echo GARCH schemas created.

echo.
echo === Step 2: Create GBM table schemas ===
python gbm_table_builder.py || goto :error
echo GBM schemas created.

cd /d "%PROJECT_ROOT%"

echo.
echo === Step 3: Populate GARCH tables from Yahoo Finance ===
python Data\Portfolio\yfinance_garch_fetcher.py || goto :error

echo.
echo === Step 4: Populate GBM tables from Yahoo Finance ===
python Data\Portfolio\yfinance_gbm_fetcher.py || goto :error

echo.
echo === Step 5: Reconcile GARCH and GBM tables ===
python Data\reconcile_portfolio_tables.py || goto :error

echo.
echo === All done! ===
exit /b 0

:error
echo.
echo ERROR: A step failed. See output above.
exit /b 1
