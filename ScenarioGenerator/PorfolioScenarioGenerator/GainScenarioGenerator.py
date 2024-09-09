import numpy as np
import warnings
from numpy.random import SFC64, SeedSequence, Generator
from PgConnection.PgConnection import PgConnection
from ScenarioGenerator.ScenarioGenerator import ScenarioGenerator


class GainScenarioGenerator(ScenarioGenerator):
    
    def __init__(self,
                 relation: str,
                 base_predicate = ''):
        self.__relation = relation
        self.__base_predicate = base_predicate
        if len(self.__base_predicate) == 0:
            self.__base_predicate = '1=1'

    def __get_info(self):
        sql_query = 'select min(ticker), min(sell_after),'\
            ' min(price), min(volatility), '\
            ' min(volatility_coeff), min(drift) from '\
            + self.__relation +\
            ' where ' + self.__base_predicate + \
            ' group by id order by id;'
        PgConnection.Execute(sql_query)
        return PgConnection.Fetch()

    def __hash(self, str):
        hashed_value = 0
        for char in str:
            hashed_value = hashed_value * 7727
            hashed_value += ord(char)
            hashed_value %= 2593697
        return hashed_value

    def generate_scenarios(self, seed, no_of_scenarios):
        info = self.__get_info()
        sell_after_dates = []
        tuple_numbers = []
        gains = []
        for _ in range(len(info)):
            gains.append([])
        tuple_number = 0
        ticker = None
        last_ticker = None
        sell_after = None
        price = None
        last_price = None
        volatility = None
        last_volatility = None
        volatility_coeff = None
        last_volatility_coeff = None
        drift = None
        last_drift = None
        for tuple in info:
            ticker, sell_after, price, volatility,\
            volatility_coeff, drift = tuple
            sell_after *= 2
            if ticker != last_ticker and last_ticker is not None:
                hashed_value = (seed + self.__hash(last_ticker))%(10**8)
                rng = Generator(SFC64(SeedSequence(hashed_value)))
                sqrt_time_intervals = []
                last_period = 0
                
                for period in sell_after_dates:
                    sqrt_time_intervals.append(
                        np.sqrt(period - last_period)
                )
                
                last_period = period
                noises = rng.normal(loc=0,
                                    scale=sqrt_time_intervals,
                                    size=(no_of_scenarios,
                                          len(sell_after_dates)))
                for scenario_number in range(no_of_scenarios):
                    curr_price = last_price
                    last_period = 0
                    counter = 0
                    for period in sell_after_dates:
                        timegap = period - last_period
                        last_period = period
                        exponent_volatility = last_volatility * last_volatility_coeff
                        exponent = (last_drift - 0.5 * exponent_volatility ** 2) * timegap
                        exponent_noise = exponent_volatility * noises[scenario_number][counter]
                        curr_price = curr_price * np.exp(exponent + exponent_noise)
                        if curr_price > 2 * last_price:
                            curr_price = 2 * last_price
                        gains[tuple_numbers[counter]].append(curr_price - last_price)
                        counter += 1
                
                tuple_numbers.clear()
                sell_after_dates.clear()
            
            sell_after_dates.append(int(sell_after))
            tuple_numbers.append(tuple_number)
            last_ticker = ticker
            last_volatility = volatility
            last_volatility_coeff = volatility_coeff
            last_drift = drift
            last_price = price
            tuple_number += 1
        
        if len(sell_after_dates) > 0:
            hashed_value = (seed + self.__hash(ticker))%(10**8)
            rng = Generator(SFC64(SeedSequence(hashed_value)))
            sqrt_time_intervals = []
            last_period = 0
            for period in sell_after_dates:
                sqrt_time_intervals.append(
                    np.sqrt(period - last_period)
                )
                last_period = period
            noises = rng.normal(loc=0,
                                scale=sqrt_time_intervals,
                                size=(no_of_scenarios,
                                len(sell_after_dates)))
            for scenario_number in range(no_of_scenarios):
                curr_price = price
                last_period = 0
                counter = 0
                for period in sell_after_dates:
                    timegap = period - last_period
                    last_period = period
                    exponent_volatility = last_volatility * last_volatility_coeff
                    exponent = (last_drift - 0.5 * exponent_volatility ** 2) * timegap
                    exponent_noise = exponent_volatility * noises[scenario_number][counter]
                    curr_price = curr_price * np.exp(exponent + exponent_noise)
                    if curr_price > 2 * last_price:
                        curr_price = 2 * last_price
                    gains[tuple_numbers[counter]].append(
                            curr_price - last_price)
                    counter += 1
            tuple_numbers.clear()
            sell_after_dates.clear()
        return gains

