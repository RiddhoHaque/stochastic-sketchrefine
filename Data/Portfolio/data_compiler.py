import os


directories = [
    'stock_market_data\\forbes2000\\csv',
    'stock_market_data\\nasdaq\\csv',
    'stock_market_data\\nyse\\csv',
    'stock_market_data\\sp500\\csv'
]

tickers = set()

for directory in directories:
    file_list = os.listdir(directory)
    for file in file_list:
        ticker = file.split('.')[0]
        if ticker not in tickers:
            tickers.add(ticker)