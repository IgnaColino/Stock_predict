# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:04:29 2019

@author: Ignacio
"""

import pandas as pd
import numpy as np
from dependent_variable import get_price_change
import feature_engineering as fe
import traceback
import datetime as dt


md = pd.to_datetime(pd.Timestamp.today().date(), utc=True) - \
    pd.Timedelta('14 days')


class dataset:
    def __init__(self, df, table, min_date=None, max_date=md, tickers=None):
        self.df = df
        self.df.sort_values(['symbol', 'date'], inplace=True)
        if table != 'cry':
            self.df = self.df.loc[self.df.adjusted_close != 0]
        else:
            self.df['adjusted_close'] = self.df.close
        self.sdf = self.df
        if max_date is not None:
            self.sdf = self.sdf.loc[self.sdf.date <= max_date]
        if min_date is not None:
            self.sdf = self.sdf.loc[self.sdf.date >= min_date -
                                    dt.timedelta(days=400)]
        if tickers is not None:
            self.sdf = self.sdf.loc[self.sdf.symbol.isin(tickers)]

    def __len__(self):
        return len(self.sdf)

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
                    # if max > 1 and min< 0
                    if (self.sdf.loc[self.sdf.symbol == ticker, col].max() > 1
                        and self.sdf.loc[self.sdf.symbol == ticker,
                                         col].min() < 0) and \
                      self.sdf.loc[self.sdf.symbol == ticker, col].std() != 0:
                        self.sdf.loc[self.sdf.symbol == ticker, col] = \
                            (self.sdf.loc[self.sdf.symbol == ticker, col] -
                             self.sdf.loc[self.sdf.symbol == ticker,
                                          col].mean()) / \
                            self.sdf.loc[self.sdf.symbol == ticker, col].std()
                    elif (self.sdf.loc[self.sdf.symbol == ticker,
                                       col].max() > 1 and
                          self.sdf.loc[self.sdf.symbol == ticker,
                                       col].min() >= 0 and
                          self.sdf.loc[self.sdf.symbol == ticker,
                                       col].std() != 0):
                        self.sdf.loc[self.sdf.symbol == ticker, col] = \
                            self.sdf.loc[self.sdf.symbol == ticker, col] / \
                            self.sdf.loc[self.sdf.symbol == ticker, col].std()
            elif col == 'woy':
                self.sdf[col] = self.sdf[col].astype(float)/52
        self.sdf.dropna(inplace=True)

    def add_target(self, days=[1], percent_change=[0.5], categorical=True):
        frames = []
        for i in self.sdf.symbol.unique():
            temp = self.sdf.loc[self.sdf.symbol == i]
            temp.reset_index(drop=True, inplace=True)
            temp = get_price_change(temp, days, percent_change)
            if categorical:
                temp.drop('price_change_percent_1', axis=1, inplace=True)
            else:
                temp.drop('price_'+str(days[0]) + '_' + str(percent_change[0]),
                          axis=1, inplace=True)
                """temp['price_change_percent_1'] = \
                    temp['price_change_percent_1']/100"""
            temp.rename(columns={temp.columns[-1]: 'target'}, inplace=True)
            temp.target = temp.target.shift(-1)
            frames.append(temp)
        self.sdf = pd.concat(frames, ignore_index=True, sort=False)
        self.sdf.rename(columns={self.sdf.columns[-1]: 'target'}, inplace=True)
        self.sdf.dropna(inplace=True)
        if categorical:
            self.sdf.replace({1: 2, 0: 1, -1: 0}, inplace=True)

    def split(self):
        self.t_tickers = pd.Series(self.sdf.symbol.unique()).sample(frac=0.8)
        self.t_tickers.to_csv('train_tickers.csv')
        self.traindata = self.sdf.loc[self.sdf.symbol.isin(self.t_tickers)]
        self.testdata = self.sdf.loc[~self.sdf.symbol.isin(self.t_tickers)]
        self.traindata.reset_index(drop=True, inplace=True)
        self.testdata.reset_index(drop=True, inplace=True)

    def prepare_dataset(self, cat=True):
        self.eng_features()
        self.add_target(categorical=cat)
        self.scale_features()
        self.split()

    def prepare_test_dataset(self, cat=True):
        self.eng_features()
        self.add_target(categorical=cat)
        self.scale_features()
        self.sdf.reset_index(drop=True, inplace=True)


class datagen:
    def __init__(self, df, gen_length=30, start_date=dt.date(2010, 1, 1),
                 end_date=dt.datetime.today().date(), test=False,
                 shuffle=False):
        self.df = df
        self.gen_length = gen_length
        self.df = self.df.loc[
            self.df.date.between(start_date -
                                 dt.timedelta(days=self.gen_length+1),
                                 end_date)]
        self.df.sort_values(['symbol', 'date'], ascending=[True, True],
                            inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.idx = 0
        self.test = test
        self.result = np.empty((0, 1), int)
        self.ticks = []
        self.shuffle = shuffle
        '''
        for sym in self.df.symbol.unique():
            self.ticks.extend(self.df.loc[self.df.symbol == sym]
                              [self.gen_length-1:].symbol.values)'''

    def __len__(self):
        datas = []
        for sym in self.df.symbol.unique():
            datas.append(max(len(self.df.loc[self.df.symbol ==
                                             sym][-self.gen_length:]) -
                             (self.gen_length - 1), 0))
        return sum(datas)

    def __next__(self):
        while True:
            try:
                if self.idx > self.df.index.max():
                    self.idx = np.random.randint(0, self.df.index.max())
                if self.shuffle:
                    self.idx = np.random.randint(0, self.df.index.max())
                temp = self.df.loc[(self.df.symbol ==
                                    self.df.loc[self.idx].symbol)
                                   & (self.df.date <=
                                      self.df.loc[self.idx].date)]
                if len(temp) >= self.gen_length:
                    temp = temp.iloc[-self.gen_length:, :]
                    self.ticks.append(temp.symbol.unique())
                    temp.drop(['symbol', 'date'], axis=1, inplace=True)
                    x, y = np.array(temp.iloc[:, :-1]),\
                        np.array(temp.iloc[-1, -1])
                    self.result = np.append(self.result,
                                            np.reshape(y, (1, 1)).astype(int),
                                            axis=0)
                    if self.test:
                        self.idx = np.random.randint(0, self.df.index.max())
                    else:
                        self.idx = self.idx+1
                    return np.reshape(x, (1, x.shape[0],
                                          x.shape[1])).astype('float32'),\
                        np.reshape(y, (1, 1)).astype(int)
                else:
                    self.idx = self.idx+self.gen_length-len(temp)
                    continue
            except Exception as e:
                print(e)
                traceback.print_exc()
