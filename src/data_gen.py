# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:04:29 2019

@author: Ignacio
"""

import pandas as pd
import numpy as np
from dependent_variable import get_price_change
import feature_engineering as fe
import datetime as dt
from tensorflow.keras.utils import Sequence
import math


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
                # if max > 1 and min< 0
                if (self.sdf.loc[:, col].max() > 1
                    and self.sdf.loc[:, col].min() < 0) and \
                    self.sdf.loc[:, col].std() != 0:
                    self.sdf.loc[:, col] = \
                        (self.sdf.loc[:, col] -
                         self.sdf.loc[:, col].mean()) / \
                        self.sdf.loc[:, col].std()
                elif (self.sdf.loc[:, col].max() > 1 and
                      self.sdf.loc[:, col].min() >= 0 and
                      self.sdf.loc[:, col].std() != 0):
                    self.sdf.loc[:, col] = \
                        self.sdf.loc[:, col] / \
                        self.sdf.loc[:, col].std()
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


class datagen(Sequence):
    def __init__(self, df, gen_length=30,
                 start_date=dt.datetime(2010, 1, 1, 0, 0, 0,
                                        tzinfo=dt.timezone.utc),
                 end_date=dt.datetime.now(dt.timezone.utc), n_classes=3,
                 test=False, shuffle=False, batch_size=32, n_channels=1):
        self.df = df
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.symbols = df.symbol.unique().tolist()
        self.gen_length = gen_length
        self.df = self.df.loc[
            self.df.date.between(start_date -
                                 dt.timedelta(days=self.gen_length+1),
                                 end_date)]
        self.df.sort_values(['symbol', 'date'], ascending=[True, True],
                            inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.test = test
        self.result = np.empty((0, 1), int)
        self.sub_list = [self.df.loc[self.df.symbol==j]
                         .index[self.gen_length:].values.tolist() for j in
                         self.df.symbol.unique()]
        self.list_IDs = [item for sublist in self.sub_list 
                         for item in sublist]
        self.ticks = []
        self.shuffle = shuffle
        self.dim = (self.gen_length, self.df.shape[1]-3)
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        datas = [max(len(self.df.loc[self.df.symbol==i,:])\
                     -self.gen_length,0) for i in self.symbols]
        return math.ceil(sum(datas)/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
                               (index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.indexes = self.indexes.tolist()


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.reshape(self.df.iloc[ID-self.gen_length:ID,:]\
                .drop(['symbol', 'date', 'target'], axis=1).to_numpy(),
                (self.dim[0],self.dim[1],1))
            # Store class
            y[i] = self.df.iloc[ID].target
        X=np.reshape(X, (self.batch_size, self.dim[0],self.dim[1]))
        return X, y
      # to_categorical(y, num_classes=self.n_classes)
