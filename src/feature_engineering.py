# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 12:43:49 2019

@author: Ignacio
"""
import numpy as np
import pandas as pd


def SMA(df, period):
    """Simple moving average"""
    df['SMA_'+str(period)] = df.adjusted_close.rolling(period).mean()


def WMA(df, com):
    """Weighted moving average"""
    df['WMA_'] = df.adjusted_close.ewm(com=com).mean()


def Momentum(df, period):
    df['Momentum_'+str(period)] = (df.adjusted_close -
                                   df.adjusted_close.shift(period)) / \
                                   df.adjusted_close.shift(period)


def Stochastic_K(df, period=14):
    df['Stochastic_K'+str(period)] = ((df.close-df.low.rolling(period).min()) /
                                      (df.high.rolling(period).max() -
                                       df.low.rolling(period).min())) * 100


def Stochastic_D(df, period=3):
    step = (df.close - df.low.rolling(14).min()) * 100 / \
           (df.high.rolling(14).max() - df.low.rolling(14).min())
    df['Stochastic_D'+str(period)] = step.rolling(period, center=False).mean()


def RSI(df, period=14):
    df2 = pd.DataFrame()
    df2['step'] = (df.adjusted_close-df.adjusted_close.shift(1)) / \
        df.adjusted_close.shift(1)
    df2['step_up'] = [i if i >= 0 else np.nan for i in df2.step]
    df2['step_down'] = [-i if i < 0 else np.nan for i in df2.step]
    df2['avg_gain'] = df2.step_up.rolling(period, min_periods=1).sum() / \
        period
    df2['avg_loss'] = df2.step_down.rolling(period, min_periods=1).sum() / \
        period
    df['RSI'+str(period)] = 100-(100/(1+df2.avg_gain/df2.avg_loss))
    df['RSI'+str(period)][:period-1] = np.nan


def Williams_R(df, period=14):
    df['Williams_R'+str(period)] = (df.high.rolling(period).max() - df.close)\
        / (df.high.rolling(period).max() - df.low.rolling(period).min()) * 100


def MACD(df):
    macd = df.adjusted_close.ewm(span=12).mean() -\
        df.adjusted_close.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    df['MACD'] = macd - signal


def Accum_distrib(df):
    mvf = ((df.close - df.low) - (df.high - df.close)) * df.volume\
        / (df.high - df.low)
    df['ADL'] = mvf.cumsum()


def CCI(df):
    typical_price = (df.high + df.low + df.close)/3
    sma20 = typical_price.rolling(20, center=False).mean()
    mean_dev = pd.Series([0 for i in range(len(typical_price))])
    for i in range(0, 20):
        mean_dev += (typical_price.shift(i)-sma20)
    mean_dev = mean_dev / 20
    df['CCI'] = (typical_price - sma20) / (0.015 * mean_dev)

"""
def T_SMA(df):
    for col in df.columns:
        if col[:3] in ['SMA', 'WMA':
            df['T_'+col]=[1 if p > s elif p==s 0 else -1 
                          for p, s in zip(df[col], df.adjusted_close)]

def T_WMA(df):
    
"""
