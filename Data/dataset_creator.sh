echo "Dropping all the data currently in the tables"
echo "Going to create the tables from scratch afterwards"
python table_builder.py
echo "Tables cleared and built"
python table_populator.py
echo "All tables populated"
echo "Going to create indexes on all these tables now...hang tight!"
python indexify.py
echo "Indexes created, all done!"