import os
import pandas as pd
import math
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
volatilities = []
drifts = []

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
                if float(row['Open']) < 0.01 or float(row['Close']) < 0.01:
                    continue
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
                open_dict[key] = float(row['Open'])
                keys.append(key)
                close_dict[key] = float(row['Close'])
                if len(keys) == 0 or keys[len(keys)-1] != key:
                    keys.append(key)
    if max_year < 2022:
        continue
    log_returns = []
    last_value = None
    for key in keys:
        if key in open_dict.keys():
            if not math.isnan(float(open_dict[key])):
                if last_value is not None:
                    log_returns.append(np.log(open_dict[key
                                        ]/(last_value)))
                last_value = open_dict[key]
        if key in close_dict.keys():
            if not math.isnan(float(close_dict[key])):
                if last_value is not None:
                    log_returns.append(np.log(close_dict[key
                                        ]/(last_value)))
                last_value = close_dict[key]
    if len(log_returns) >= 730 and \
        last_value is not None:
        alpha = np.mean(log_returns[-730:])
        volatility = np.std(log_returns[-730:])
        drift = alpha + 0.5 * volatility * volatility
        if math.isnan(last_value) or \
            math.isnan(volatility) or \
            math.isnan(drift):
            continue
        portfolio_dict = {'ticker': ticker, 'price': round(last_value, 6),
                          'volatility': round(volatility, 6),
                          'drift': round(drift, 6)}
        if len(portfolio_dicts) == 0 or \
            portfolio_dicts[-1] != portfolio_dict:
            portfolio_dicts.append(portfolio_dict)
            volatilities.append(portfolio_dict['volatility'])
            drifts.append(portfolio_dict['drift'])

average_volatility = np.average(volatilities)
average_drift = np.average(drifts)
stddev_volatility = np.std(volatilities)
stddev_drift = np.std(drifts)

cleaned_portfolio_dicts = []

for portfolio_dict in portfolio_dicts:
    volatility = portfolio_dict['volatility']
    drift = portfolio_dict['drift']

    if drift >= average_drift - 3 * stddev_drift and \
        drift <= average_drift + 3 * stddev_drift and \
            volatility >= average_volatility - 3 * stddev_volatility and \
                volatility <= average_volatility + 3 * stddev_volatility:
        cleaned_portfolio_dicts.append(
            portfolio_dict
        )
    else:
        print('Excluding', portfolio_dict['ticker'])

portfolio_df = pd.DataFrame(cleaned_portfolio_dicts)
portfolio_df.to_csv('portfolio.csv')