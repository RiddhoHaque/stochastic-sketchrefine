import random
import numpy as np
from numpy.random import SeedSequence
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

TPCH_FILE = 'lineitem.tbl'
PORTFOLIO_FILE = 'portfolio.csv'

TPCH_TABLE_NAME = 'Lineitem'
PORTFOLIO_TABLE_NAME = 'Stock_Investments'

TPCH_TUPLE_VARIANT_SUBSTRING = ''
PORTFOLIO_TUPLE_VARIANT_SUBSTRING = ''

TPCH_VARIANCE_VARIANT_SUBSTRING = 'Variance'
PORTFOLIO_VARIANCE_VARIANT_SUBSTRING = 'Volatility'

TPCH_LAMBDA_VARIANT_SUBSTRING = 'Lambda'
PORTFOLIO_LAMBDA_VARIANT_SUBSTRING = 'Volatility_Lambda'

TPCH_TUPLE_VARIATION_SUBSTRINGS = ['20000', '60000', '120000', 
                                  '300000', '450000', '600000',
                                  '1200000', '3000000', '4500000',
                                  '6000000']

TPCH_TUPLE_VARIATIONS = [20000, 60000, 120000, 300000, 450000, 
                        600000, 1200000, 3000000, 4500000,
                        6000000]

PORTFOLIO_TUPLE_VARIATION_SUBSTRINGS = ['90', '45', '30', '15',
                                        '9', '3', '1', 'half']

PORTFOLIO_TUPLE_VARIATIONS = [90, 45, 30, 15,
                              9, 3, 1, 0.5]

TPCH_VARIANCE_VARIATION_SUBSTRINGS = ['1x', '2x', '5x', '8x',
                                      '10x', '13x', '17x', '20x']

TPCH_VARIANCE_VARIATIONS = [1, 2, 5, 8,
                            10, 13, 17, 20]

PORTFOLIO_VARIANCE_VARIATION_SUBSTRINGS = ['1x', '2x', '5x', '8x',
                                           '10x', '13x', '17x', '20x']

PORTFOLIO_VARIANCE_VARIATIONS = [1, 2, 5, 8, 10,
                                 13, 17, 20]

TPCH_LAMBDA_VARIATION_SUBSTRINGS = ['halfx', '1x', '2x', '3x',
                                    '4x', '5x']

TPCH_LAMBDA_VARIATIONS = [0.5, 1, 2, 3, 4, 5]

PORTFOLIO_LAMBDA_VARIATION_SUBSTRINGS = ['halfx', '1x', '2x',
                                         '3x', '4x', '5x']

PORTFOLIO_LAMBDA_VARIATIONS = [0.5, 1, 2, 3, 4, 5]

tpch_attributes = [
    'id',
    'orderkey',
    'partkey',
    'suppkey',
    'linenumber',
    'quantity',
    'quantity_mean',
    'quantity_variance',
    'quantity_variance_coeff',
    'price',
    'price_mean',
    'price_variance',
    'price_variance_coeff',
    'tax'
]

lineitem_schema_index = {
    'orderkey' : 0,
    'partkey' : 1,
    'suppkey' : 2,
    'linenumber' : 3,
    'quantity' : 4,
    'price' : 5,
    'tax' : 7
 }

portfolio_attributes = [
    'id',
    'ticker',
    'sell_after',
    'price',
    'volatility',
    'volatility_coeff',
    'drift'
]

portfolio_index = {
    'ticker' : 1,
    'price' : 2,
    'volatility' : 3,
    'drift' : 4
}

INIT_SEED = SeedSequence(2342123)
rng = np.random.default_rng(INIT_SEED)

def create_portfolio_tuple_variant_datasets(
        interval, interval_string, cursor
):
    table_name = PORTFOLIO_TABLE_NAME + '_' + interval_string
    is_first_line = True
    row_number = 0
    for line in open(PORTFOLIO_FILE, 'r').readlines():
        if is_first_line:
            is_first_line = False
        else:
            values = line.split(',')
            sell_after = 0
            while sell_after < 730:
                sell_after += interval
                tuple = dict()
                for attribute in portfolio_attributes:
                    if attribute == 'id':
                        tuple[attribute] = str(row_number)
                        row_number += 1
                    if attribute in portfolio_index:
                        if attribute == 'ticker':
                            tuple[attribute] = "'" + \
                                str(values[portfolio_index[attribute]])\
                                + "'"
                        else:
                            tuple[attribute] = str(values[
                                portfolio_index[attribute]])
                    if attribute == 'sell_after':
                        tuple[attribute] = str(sell_after)
                    if attribute == 'volatility_coeff':
                        tuple[attribute] = str(1)
                values_string = ''
                for attribute in tuple:
                    if len(values_string) > 0:
                        values_string += ', '
                    values_string += tuple[attribute]
                sql_command = "INSERT INTO " + table_name + " VALUES (" + \
                values_string + ");"
                execute_sql(cursor, sql_command)
    print('Populated', table_name)

def create_portfolio_volatility_variant_datasets(
        volatility_coeff, volatility_coeff_string, cursor
):
    table_name = PORTFOLIO_TABLE_NAME + '_' + PORTFOLIO_VARIANCE_VARIANT_SUBSTRING \
        + '_' + volatility_coeff_string
    is_first_line = True
    row_number = 0
    interval = 90
    for line in open(PORTFOLIO_FILE, 'r').readlines():
        if is_first_line:
            is_first_line = False
        else:
            values = line.split(',')
            sell_after = 0
            while sell_after < 730:
                sell_after += interval
                tuple = dict()
                for attribute in portfolio_attributes:
                    if attribute == 'id':
                        tuple[attribute] = str(row_number)
                        row_number += 1
                    if attribute in portfolio_index:
                        if attribute == 'ticker':
                            tuple[attribute] = "'" + \
                                str(values[portfolio_index[attribute]])\
                                + "'"
                        else:
                            tuple[attribute] = str(values[
                                portfolio_index[attribute]])
                    if attribute == 'sell_after':
                        tuple[attribute] = str(sell_after)
                    if attribute == 'volatility_coeff':
                        tuple[attribute] = str(volatility_coeff)
                values_string = ''
                for attribute in tuple:
                    if len(values_string) > 0:
                        values_string += ', '
                    values_string += tuple[attribute]
                sql_command = "INSERT INTO " + table_name + " VALUES (" + \
                values_string + ");"
                execute_sql(cursor, sql_command)
    print('Populated', table_name)

def create_portfolio_volatility_coeff_variant_datasets(
        volatility_coeff, volatility_coeff_string, cursor
):
    table_name = PORTFOLIO_TABLE_NAME + '_' + PORTFOLIO_LAMBDA_VARIANT_SUBSTRING \
        + '_' + volatility_coeff_string
    volatility_coeffs = rng.exponential(scale = (1/volatility_coeff),
                                              size = 3457*730*2)
    is_first_line = True
    row_number = 0
    interval = 0.5
    for line in open(PORTFOLIO_FILE, 'r').readlines():
        if is_first_line:
            is_first_line = False
        else:
            values = line.split(',')
            sell_after = 0
            while sell_after < 730:
                sell_after += interval
                tuple = dict()
                for attribute in portfolio_attributes:
                    if attribute == 'id':
                        tuple[attribute] = str(row_number)
                    if attribute in portfolio_index:
                        if attribute == 'ticker':
                            tuple[attribute] = "'" + \
                                str(values[portfolio_index[attribute]])\
                                + "'"
                        else:
                            tuple[attribute] = str(values[
                                portfolio_index[attribute]])
                    if attribute == 'sell_after':
                        tuple[attribute] = str(sell_after)
                    if attribute == 'volatility_coeff':
                        tuple[attribute] = str(volatility_coeffs[row_number])
                values_string = ''
                for attribute in tuple:
                    if len(values_string) > 0:
                        values_string += ', '
                    values_string += tuple[attribute]
                row_number += 1
                sql_command = "INSERT INTO " + table_name + " VALUES (" + \
                values_string + ");"
                execute_sql(cursor, sql_command)
    print('Populated', table_name)

def create_lineitem_tuple_variant_datasets(
        no_of_tuples, total_tuples, cursor):
    row_numbers = [i for i in range(total_tuples)]
    table_name = TPCH_TABLE_NAME + '_' + str(no_of_tuples)
    rng.shuffle(row_numbers)
    selected_rows = [row_numbers[i]\
                      for i in range(no_of_tuples)]
    selected_rows.sort()
    row_number = 0
    rows_selected = 0
    next_selected_row = selected_rows[0]
    quantity_means = rng.normal(loc=0, scale=1, size=no_of_tuples)
    quantity_variances = rng.exponential(scale=0.5, size=no_of_tuples)
    price_means = rng.normal(loc=0, scale=1, size=no_of_tuples)
    price_variances = rng.exponential(scale=0.5, size=no_of_tuples)
    for line in open(TPCH_FILE, 'r').readlines():
        if row_number == next_selected_row:
            values = line.split("|")
            tuple = dict()
            for attribute in tpch_attributes:
                if attribute == 'id':
                    tuple[attribute] = str(rows_selected)
                if attribute in lineitem_schema_index:
                    tuple[attribute] = values[
                        lineitem_schema_index[
                            attribute]]
                if attribute == 'quantity_mean':
                    tuple[attribute] = str(round(quantity_means[rows_selected], 6))
                if attribute == 'quantity_variance':
                    tuple[attribute] = str(round(quantity_variances[rows_selected], 6))
                if attribute == 'quantity_variance_coeff':
                    tuple[attribute] = str(1)
                if attribute == 'price_mean':
                    tuple[attribute] = str(round(price_means[rows_selected], 6))
                if attribute == 'price_variance':
                    tuple[attribute] = str(round(price_variances[rows_selected], 6))
                if attribute == 'price_variance_coeff':
                    tuple[attribute] = str(1)
            values_string = ''
            for attribute in tuple:
                if len(values_string) > 0:
                    values_string += ', '
                values_string += tuple[attribute]
            sql_command = "INSERT INTO " + table_name + " VALUES (" + \
                values_string + ");"
            execute_sql(cursor, sql_command)
            rows_selected += 1
            if rows_selected == len(selected_rows):
                break
            next_selected_row = selected_rows[rows_selected]
        row_number += 1
    print('Populated', table_name)

def create_lineitem_variance_variant_datasets(
        variance_coefficient, variance_substring, total_tuples, cursor):
    no_of_tuples = 20000
    row_numbers = [i for i in range(total_tuples)]
    table_name = TPCH_TABLE_NAME + '_' + TPCH_VARIANCE_VARIANT_SUBSTRING + \
        '_' + variance_substring
    rng.shuffle(row_numbers)
    selected_rows = [row_numbers[i]\
                      for i in range(no_of_tuples)]
    selected_rows.sort()
    row_number = 0
    rows_selected = 0
    next_selected_row = selected_rows[0]
    quantity_means = rng.normal(loc=0, scale=1, size=no_of_tuples)
    quantity_variances = rng.exponential(scale=0.5, size=no_of_tuples)
    price_means = rng.normal(loc=0, scale=1, size=no_of_tuples)
    price_variances = rng.exponential(scale=0.5, size=no_of_tuples)
    for line in open(TPCH_FILE, 'r').readlines():
        if row_number == next_selected_row:
            values = line.split("|")
            tuple = dict()
            for attribute in tpch_attributes:
                if attribute == 'id':
                    tuple['id'] = str(rows_selected)
                if attribute in lineitem_schema_index:
                    tuple[attribute] = values[
                        lineitem_schema_index[
                            attribute]]
                if attribute == 'quantity_mean':
                    tuple[attribute] = str(round(quantity_means[rows_selected], 6))
                if attribute == 'quantity_variance':
                    tuple[attribute] = str(round(quantity_variances[rows_selected], 6))
                if attribute == 'quantity_variance_coeff':
                    tuple[attribute] = str(variance_coefficient)
                if attribute == 'price_mean':
                    tuple[attribute] = str(round(price_means[rows_selected], 6))
                if attribute == 'price_variance':
                    tuple[attribute] = str(round(price_variances[rows_selected], 6))
                if attribute == 'price_variance_coeff':
                    tuple[attribute] = str(variance_coefficient)
            values_string = ''
            for attribute in tuple:
                if len(values_string) > 0:
                    values_string += ', '
                values_string += tuple[attribute]
            sql_command = "INSERT INTO " + table_name + " VALUES (" + \
                values_string + ");"
            execute_sql(cursor, sql_command)
            rows_selected += 1
            if rows_selected == len(selected_rows):
                break
            next_selected_row = selected_rows[rows_selected]
        row_number += 1
    print('Populated', table_name)


def create_lineitem_lambda_variant_datasets(
        _lambda, lambda_string, cursor):
    total_tuples = 6000000
    no_of_tuples = total_tuples
    row_numbers = [i for i in range(total_tuples)]
    table_name = TPCH_TABLE_NAME + '_' + TPCH_LAMBDA_VARIANT_SUBSTRING + \
                '_' + lambda_string
    
    rng.shuffle(row_numbers)
    selected_rows = [row_numbers[i]\
                      for i in range(no_of_tuples)]
    selected_rows.sort()
    row_number = 0
    rows_selected = 0
    next_selected_row = selected_rows[0]
    quantity_means = rng.normal(loc=0, scale=1, size=no_of_tuples)
    quantity_variances = rng.exponential(scale=0.5, size=no_of_tuples)
    quantity_variance_coeffs = rng.exponential(scale=(1/_lambda), size=no_of_tuples)
    price_means = rng.normal(loc=0, scale=1, size=no_of_tuples)
    price_variances = rng.exponential(scale=0.5, size=no_of_tuples)
    price_variance_coeffs = rng.exponential(scale=(1/_lambda), size=no_of_tuples)
    
    for line in open(TPCH_FILE, 'r').readlines():
        if row_number == next_selected_row:
            values = line.split("|")
            tuple = dict()
            for attribute in tpch_attributes:
                if attribute == 'id':
                    tuple['id'] = str(rows_selected)
                if attribute in lineitem_schema_index:
                    tuple[attribute] = values[
                        lineitem_schema_index[
                            attribute]]
                if attribute == 'quantity_mean':
                    tuple[attribute] = str(round(quantity_means[rows_selected], 6))
                if attribute == 'quantity_variance':
                    tuple[attribute] = str(round(quantity_variances[rows_selected], 6))
                if attribute == 'quantity_variance_coeff':
                    tuple[attribute] = str(round(quantity_variance_coeffs[rows_selected], 6))
                if attribute == 'price_mean':
                    tuple[attribute] = str(round(price_means[rows_selected], 6))
                if attribute == 'price_variance':
                    tuple[attribute] = str(round(price_variances[rows_selected], 6))
                if attribute == 'price_variance_coeff':
                    tuple[attribute] = str(round(price_variance_coeffs[rows_selected], 6))
            values_string = ''
            for attribute in tuple:
                if len(values_string) > 0:
                    values_string += ', '
                values_string += tuple[attribute]
            sql_command = "INSERT INTO " + table_name + " VALUES (" + \
                values_string + ");"
            execute_sql(cursor, sql_command)
            rows_selected += 1
            if rows_selected == len(selected_rows):
                break
            next_selected_row = selected_rows[rows_selected]
        row_number += 1
    print('Populated', table_name)


conn = connect_to_database()
cursor = get_cursor(conn)

for no_of_tuples in TPCH_TUPLE_VARIATIONS:
    create_lineitem_tuple_variant_datasets(
        no_of_tuples=no_of_tuples, total_tuples=6000000,
        cursor=cursor)
print('Populated lineitem relations with different number of tuples')

for _ in range(len(TPCH_VARIANCE_VARIATIONS)):
    create_lineitem_variance_variant_datasets(
        variance_coefficient=TPCH_VARIANCE_VARIATIONS[_],
        variance_substring=TPCH_VARIANCE_VARIATION_SUBSTRINGS[_],
        total_tuples=6000000,
        cursor=cursor)
print('Populated lineitem relations with different fixed variance coefficients')


for _ in range(len(TPCH_LAMBDA_VARIATIONS)):
    create_lineitem_lambda_variant_datasets(
        _lambda = TPCH_LAMBDA_VARIATIONS[_],
        lambda_string = TPCH_LAMBDA_VARIATION_SUBSTRINGS[_],
        cursor=cursor
    )

print('Populated lineitem relations with different variance coefficients')

for _ in range(len(PORTFOLIO_TUPLE_VARIATIONS)):
    create_portfolio_tuple_variant_datasets(
        PORTFOLIO_TUPLE_VARIATIONS[_],
        PORTFOLIO_TUPLE_VARIATION_SUBSTRINGS[_],
        cursor
    )

print('Populated portfolio relations with different number of tuples')

for _ in range(len(PORTFOLIO_VARIANCE_VARIATIONS)):
    create_portfolio_volatility_variant_datasets(
        PORTFOLIO_VARIANCE_VARIATIONS[_],
        PORTFOLIO_VARIANCE_VARIATION_SUBSTRINGS[_],
        cursor
    )

print('Populated portfolio relations with different volatilities')


for _ in range(len(PORTFOLIO_LAMBDA_VARIATIONS)):
    create_portfolio_volatility_coeff_variant_datasets(
        PORTFOLIO_LAMBDA_VARIATIONS[_],
        PORTFOLIO_LAMBDA_VARIATION_SUBSTRINGS[_],
        cursor
    )

print('Populated portfolio relations with different scale factor for volatility coefficients')

conn.commit()
cursor.close()
conn.close()
