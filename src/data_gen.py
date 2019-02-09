# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:04:29 2019

@author: Ignacio
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import ast
import os
from dependent_variable import get_price_change


class dataset:
    def __init__(self, table):
        self.security = ast.literal_eval(os.getenv('PSQL_USER'))
        self.conn = create_engine("postgresql://" +
                                  self.security['user'] + ":" +
                                  self.security['password'] +
                                  "@localhost/stock_data")
        self.df = pd.read_sql_table(table, self.conn)
        self.df.sort_values(['symbol', 'date'], inplace=True)
        self.df = self.df.loc[self.df.adjusted_close != 0]

    def add_target(self):
        frames = []
        for i in self.df.symbol.unique():
            temp = self.df.loc[self.df.symbol == i]
            temp.reset_index(drop=True, inplace=True)
            temp = get_price_change(temp, [1], [0.5])
            temp.drop('price_change_percent_1', axis=1, inplace=True)
            temp.rename(columns={temp.columns[-1]: 'target'}, inplace=True)
            temp.target = temp.target.shift(-1)
            frames.append(temp)
        self.df = pd.concat(frames, ignore_index=True, sort=False)
        self.df.rename(columns={self.df.columns[-1]: 'target'}, inplace=True)
        self.df.dropna(inplace=True)
        self.df.replace({1: 2, 0: 1, -1: 0}, inplace=True)

    def split(self):
        self.t_tickers = pd.Series(self.df.symbol.unique()).sample(frac=0.8)
        self.traindata = self.df.loc[self.df.symbol.isin(self.t_tickers)]
        self.testdata = self.df.loc[~self.df.symbol.isin(self.t_tickers)]
        self.traindata.reset_index(drop=True, inplace=True)
        self.testdata.reset_index(drop=True, inplace=True)


class datagen:
    def __init__(self, df, gen_length=30):
        self.df = df
        self.gen_length = gen_length
        self.idx = np.random.randint(0, len(self.df))

    def __next__(self):
        while True:
            temp = self.df.loc[(self.df.symbol == self.df.loc[self.idx].symbol)
                               & (self.df.date <= self.df.loc[self.idx].date)]
            self.idx = self.idx+1
            if len(temp) >= self.gen_length:
                temp = temp.iloc[-self.gen_length:, :]
                temp.drop(['symbol', 'date'], axis=1, inplace=True)
                x, y = np.array(temp.iloc[:, :-1]), np.array(temp.iloc[-1, -1])
                self.idx = np.random.randint(0, len(self.df))
                return np.reshape(x, (1, x.shape[0],
                                      x.shape[1])).astype('float32'),\
                    np.reshape(y, (1, 1)).astype(int)
            else:
                continue


if __name__ == '__main__':
    pass
    """gen = datagen('asx')
    df2 = next(gen)
    df3 = next(gen)"""
