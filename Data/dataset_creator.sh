echo "Dropping all the data currently in the tables"
echo "Going to create the tables from scratch afterwards"
python table_builder.py
echo "Tables cleared and built"
python table_populator.py
echo "All tables populated"