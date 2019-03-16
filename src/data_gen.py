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
import feature_engineering as fe
import logging
import traceback


logging.basicConfig(filename='logfile.log', level=logging.INFO)
"""to do: scale features and feature engineering"""


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
        self.sdf = self.df

    def eng_features(self):
        frames = []
        for ticker in self.sdf.symbol.unique():
            temp = self.sdf.loc[self.sdf.symbol == ticker]
            for i in [5, 10, 20, 60, 100, 200]:
                fe.SMA(temp, i)
                fe.WMA(temp, i)
                fe.Momentum(temp, i)
            fe.Stochastic_K(temp)
            fe.Stochastic_D(temp)
            fe.RSI(temp)
            fe.Williams_R(temp)
            fe.MACD(temp)
            fe.Accum_distrib(temp)
            frames.append(temp)
        self.sdf = pd.concat(frames, ignore_index=True, sort=False)

    def scale_features(self):
        """Scale non-binary features"""
        for col in self.sdf.columns:
            if col != 'woy' and col != 'symbol' and col != 'date' and \
              col != 'target':
                self.sdf.loc[col] = self.sdf[col].astype(float)
                for ticker in self.sdf.symbol.unique():
                    if (self.sdf.loc[self.sdf.symbol == ticker, col].max() > 1
                        or self.sdf.loc[self.sdf.symbol == ticker,
                                        col].min() < 0) and \
                      self.sdf.loc[self.sdf.symbol == ticker, col].std() != 0:
                        self.sdf.loc[self.sdf.symbol == ticker, col] = \
                            (self.sdf.loc[self.sdf.symbol == ticker, col] -
                             self.sdf.loc[self.sdf.symbol == ticker,
                                          col].mean()) / \
                            self.sdf.loc[self.sdf.symbol == ticker, col].std()
            elif col == 'woy':
                self.sdf[col] = self.sdf[col].astype(float)/52
        self.sdf.dropna(inplace=True)

    def add_target(self):
        frames = []
        for i in self.sdf.symbol.unique():
            temp = self.sdf.loc[self.sdf.symbol == i]
            temp.reset_index(drop=True, inplace=True)
            temp = get_price_change(temp, [1], [0.5])
            temp.drop('price_change_percent_1', axis=1, inplace=True)
            temp.rename(columns={temp.columns[-1]: 'target'}, inplace=True)
            temp.target = temp.target.shift(-1)
            frames.append(temp)
        self.sdf = pd.concat(frames, ignore_index=True, sort=False)
        self.sdf.rename(columns={self.sdf.columns[-1]: 'target'}, inplace=True)
        self.sdf.dropna(inplace=True)
        self.sdf.replace({1: 2, 0: 1, -1: 0}, inplace=True)

    def split(self):
        self.t_tickers = pd.Series(self.sdf.symbol.unique()).sample(frac=0.8)
        self.t_tickers.to_csv('train_tickers.csv')
        self.traindata = self.sdf.loc[self.sdf.symbol.isin(self.t_tickers)]
        self.testdata = self.sdf.loc[~self.sdf.symbol.isin(self.t_tickers)]
        self.traindata.reset_index(drop=True, inplace=True)
        self.testdata.reset_index(drop=True, inplace=True)

    def prepare_dataset(self):
        self.eng_features()
        self.add_target()
        self.scale_features()
        self.split()


class datagen:
    def __init__(self, df, gen_length=30):
        self.df = df
        self.df.reset_index(drop=True, inplace=True)
        self.gen_length = gen_length
        self.idx = np.random.randint(0, self.df.index.max())

    def __next__(self):
        while True:
            try:
                temp = self.df.loc[(self.df.symbol ==
                                    self.df.loc[self.idx].symbol)
                                   & (self.df.date <=
                                      self.df.loc[self.idx].date)]
                self.idx = self.idx+1
                if self.idx > self.df.index.max():
                    self.idx = np.random.randint(0, self.df.index.max())
                if len(temp) >= self.gen_length:
                    temp = temp.iloc[-self.gen_length:, :]
                    temp.drop(['symbol', 'date'], axis=1, inplace=True)
                    x, y = np.array(temp.iloc[:, :-1]),\
                        np.array(temp.iloc[-1, -1])
                    logging.info(f"{self.df.loc[self.idx-1].symbol} stock with\
                                    {len(temp)}")
                    return np.reshape(x, (1, x.shape[0],
                                          x.shape[1])).astype('float32'),\
                        np.reshape(y, (1, 1)).astype(int)
                else:
                    logging.info(f"{self.df.loc[self.idx-1].symbol} stock with\
                                    {len(temp)}")
                    continue
            except Exception as e:
                traceback.print_exc()
