from configparser import ConfigParser
import psycopg2

# Function copied from
# gist.github.com/ZacharyMcGuire/d81aa85409594a007fdf80e9fa9b329e

def config(filename='database.ini', section = 'postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    db_config = {}

    if section in parser:
        for key in parser[section]:
            db_config[key] = parser[section][key]

    return db_config

def connect_to_database():
    db_config = config()
    return psycopg2.connect(
        dbname = db_config['dbname'],
        user = db_config['user'],
        host = db_config['host'],
        password = db_config['password'],
        port = db_config['port']
    )

def get_cursor(conn):
    return conn.cursor()

def execute_sql(cursor, sql):
    cursor.execute(sql)

conn = connect_to_database()
cursor = get_cursor(conn)
file = open("indexify.sql", "r")
lines = file.readlines()
sql_command = ''
for line in lines:
    sql_command += ' '
    sql_command += line
execute_sql(cursor, sql_command)
conn.commit()
cursor.close()
conn.close()
