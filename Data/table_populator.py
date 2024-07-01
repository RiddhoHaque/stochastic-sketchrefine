import random
import numpy as np

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

TPC_TUPLE_VARIATIONS = [20000, 60000, 120000, 300000, 450000, 
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

class RandomSeedGenerator:
    INIT_RANDOM_SEED = 2342123
    CURRENT_SEED = INIT_RANDOM_SEED
    @staticmethod
    def getNextSeed():
        random.seed(CURRENT_SEED)
        CURRENT_SEED = random.random()

def create_lineitem_tuple_variant_datsets(
        no_of_tuples, total_tuples):
    row_numbers = [i for i in range(total_tuples)]
    random.shuffle(row_numbers)
    selected_rows = [row_numbers[i]\
                      for i in range(no_of_tuples)]
    selected_rows.sort()
    row_number = 0
    rows_selected = 0
    next_selected_row = selected_rows[0]
    for line in open(TPCH_FILE, 'r').readlines():
        if row_number == next_selected_row:
            values = line.split("|")
            tuple = dict()
            for attribute in tpch_attributes:
                if attribute == 'id':
                    tuple['id'] = rows_selected
                if attribute in lineitem_schema_index:
                    tuple[attribute] = values[
                        lineitem_schema_index[
                            attribute]]
                if attribute == 'quantity_mean':

        row_number += 1



