# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:18:50 2019

@author: Ignacio Colino
"""
import time
import pandas as pd
import requests
import os
"""This module has the function to get the daily price data for different
markets."""


# Get the adjusted Data
def get_adjusted_data(market='ASX', num_stocks=None, tickers=None):
    """This function outputs a csv file with the historical daily price data
    for the corresponding market. The stock codes are picked up from the csv
    files in the folder"""
    try:
        # Parameters for API request
        stock_list = {'ASX': '20180801-asx200.csv',
                      'SNP': '20180922_SP500_list.csv'}
        query_param = {'ASX': 'ASX:', 'SNP': ''}
        symbol_list = pd.read_csv(stock_list[market], header=1)
        names = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume',
                 'dividend_amount', 'split_coefficient', 'symbol', 'date']
        data = pd.DataFrame(columns=['Symbol'])
        url = "https://www.alphavantage.co/query"
        if num_stocks is not None:
            symbol_list = symbol_list.sample(num_stocks)
        if tickers is not None:
            symbol_list = pd.DataFrame(tickers)
        # Loop through stock list and concatenate
        for code in symbol_list.iloc[:, 0]:
            if query_param[market]+code not in data['Symbol'].unique():
                # query structure
                para = {"function": "TIME_SERIES_DAILY_ADJUSTED",
                        "symbol": query_param[market]+code,
                        "outputsize": "full",
                        "apikey": os.getenv('ALPHA_VANTAGE')}
                page = requests.get(url, params=para)
                time.sleep(13)  # 5 requests per minute allowed
                print(code, end=" ")
                data2 = pd.DataFrame.from_dict(page.json()
                                               ['Time Series (Daily)'],
                                               orient='index', dtype=float)
                data2['Symbol'] = page.json()['Meta Data']['2. Symbol']
                data2.index = pd.to_datetime(data2.index)
                data2.reset_index(level=0, inplace=True)
                data2['index'] = data2['index'].apply(lambda x: x.date())
                data = pd.concat([data, data2], axis=0, ignore_index=True,
                                 sort=True)

        # Print Summary and export to csv
        print('Nbr of datapoints:', len(data))
        print('Nbr of Companies:', len(data['Symbol'].unique()))
        print('Aprox years of data per company:',
              round(len(data)/len(data['Symbol'].unique())/240, 2))

        data.rename(columns={i: j for i, j in zip(data.columns, names)},
                    inplace=True)
        # data.to_csv(market+'_adjusted_data.csv', index=False)
    except Exception as error:
        print(error)
    finally:
        return data


if __name__ == "__main__":
    data = get_adjusted_data(market='SNP', num_stocks=2)
