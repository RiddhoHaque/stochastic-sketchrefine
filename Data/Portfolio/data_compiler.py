import os
import pandas as pd
import numpy as np


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

portfolio_dicts = []

for ticker in tickers:
    open_dict = dict()
    close_dict = dict()
    keys = []
    max_year = 0
    for directory in directories:
        file_to_search = ticker + '.csv' 
        if file_to_search in os.listdir(directory):
            file_name = directory + '\\' + file_to_search
            timeline_df = pd.read_csv(file_name)
            for _, row in timeline_df.iterrows():
                if len(row['Date'].split('-')) < 3:
                    print('Directory:', directory,
                          'Ticker:', ticker,
                          'Date:', row['Date'])
                year = int(row['Date'].split('-')[2])
                if year > max_year:
                    max_year = year
                month = int(row['Date'].split('-')[1])
                day = int(row['Date'].split('-')[0])
                key = (year, month, day)
                if float(row['Open']) >= 0.01:
                    open_dict[key] = float(row['Open'])
                    keys.append(key)
                if float(row['Close']) >= 0.01:
                    close_dict[key] = float(row['Close'])
                    if len(keys) == 0 or keys[len(keys)-1] != key:
                        keys.append(key)
    if max_year < 2022:
        continue
    log_returns = []
    last_value = None
    for key in keys:
        if key in open_dict.keys():
            if last_value is not None:
                log_returns.append(np.log(open_dict[key
                                    ]/(last_value)))
            last_value = open_dict[key]
        if key in close_dict.keys():
            if last_value is not None:
                log_returns.append(np.log(close_dict[key
                                    ]/(last_value)))
            last_value = close_dict[key]
    if len(log_returns) >= 1:
        alpha = np.mean(log_returns)
        volatility = np.std(log_returns)
        drift = alpha + 0.5 * volatility * volatility
        portfolio_dict = {'ticker': ticker, 'price': round(last_value, 6),
                          'volatility': round(volatility, 6),
                          'drift': round(drift, 6)}
    portfolio_dicts.append(portfolio_dict)

portfolio_df = pd.DataFrame(portfolio_dicts)
portfolio_df.to_csv('portfolio.csv')