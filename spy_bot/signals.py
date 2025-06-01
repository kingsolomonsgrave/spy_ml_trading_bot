#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:36:25 2025

@author: maurihall
"""
import pandas as pd
import numpy as np

def conditions_vectorized(i, HLOCV_EMA, larger_body=True):
    c1 = i - 2
    c2 = i - 1
    c3 = i
    # High, Low, Open, Close, Volume, EMA20
    highs = HLOCV_EMA[0]
    lows = HLOCV_EMA[1]
    opens = HLOCV_EMA[2]
    closes = HLOCV_EMA[3]
    volumes = HLOCV_EMA[4]
    ema20 = HLOCV_EMA[5]
    
    if larger_body:
        c3_body = abs(opens[i] - closes[i])
        c2_body = abs(opens[i-1] - closes[i-1])
        c3_body_larger_than_c2_body = c3_body > c2_body
    else:
        c3_body_larger_than_c2_body = True

    return (
        closes[c1] < ema20[c1] and
        closes[c2] > ema20[c2] and
        closes[c3] > ema20[c3] and
        closes[c3] > highs[c2] and
        ((highs[c2] - closes[c2]) / closes[c2]) < 0.05 and
        ((highs[c3] - closes[c3]) / closes[c3]) < 0.01 and
        volumes[c3] > volumes[c2] and c3_body_larger_than_c2_body )
    

def conditions(c1, c2, c3, tail1 = 0.05, tail2 = 0.01, larger_body= True, df = None):
    c3_body = abs(c3["Open"] - c3["Close"])
    c2_body = abs(c2["Open"] - c2["Close"])

    cond1 = c1['Close'] < c1['ema20'] ## CHANGED TO Close
    cond2 = c2['Close'] > c2['ema20']
    cond3 = c3['Close'] > c3['ema20']
    cond4 = c3['Close'] > c2['High']
    cond5 = ((c2['High'] - c2['Close']) / c2['Close']) < tail1
    cond6 = ((c3['High'] - c3['Close']) / c3['Close']) < tail2
    cond7 = c3["Volume"] > c2["Volume"] if c3["Volume"] and c2["Volume"] else False
    if larger_body:
        cond8 = c3_body > c2_body
    else:
        cond8 = True
    #if df is not None:
       # cond9 = all(df[i-12:i-1] < c3["Close"])
       # print(f'cond9 is {cond9}')
    #else:
        #cond9 = True
    if cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8: # and cond9:
        return True

def candle_conditions(c1, c2, c3, tail1=0.05, tail2=0.01, larger_body=True):
    c3_body = abs(c3["Open"] - c3["Close"])
    c2_body = abs(c2["Open"] - c2["Close"])
    return all([
        c1['Close'] < c1['ema20'],
        c2['Close'] > c2['ema20'],
        c3['Close'] > c3['ema20'],
        c3['Close'] > c2['High'],
        ((c2['High'] - c2['Close']) / c2['Close']) < tail1,
        ((c3['High'] - c3['Close']) / c3['Close']) < tail2,
        c3["Volume"] > c2["Volume"] if c3["Volume"] and c2["Volume"] else False,
        c3_body > c2_body if larger_body else True
    ])

def should_take_trade(row1, row2, row3, df_slice, model, add = True):
    volatility = df_slice['High'].std()
    momentum = row3['Close'] - row1['Close']
    vol_change = (row3['Volume'] - row2['Volume']) / row2['Volume']
    #print("row2 ", row2)
    ema_slope = row3['ema20'] - row2['ema20']
    percent_change = (row3['Close'] - row2['Close'])/row2['Close']
    X_live = pd.DataFrame([[volatility, momentum, vol_change, ema_slope, percent_change]], columns=['volatility', 'momentum', 'vol_change', 'ema_slope', 'percent_change'])
    return model.predict(X_live)[0] == 1
    
    
   
def should_take_trade_vectorized(i, HLOCV_EMA, model):
    """
    Fast version of should_take_trade using numpy arrays.
    Assumes you're at index i and have 20 bars of history.
    """
    # Feature engineering (same as your earlier feature_list logic)
    high = HLOCV_EMA[0]
    low = HLOCV_EMA[1]
    open_ = HLOCV_EMA[2]
    close =HLOCV_EMA[3]
    volume = HLOCV_EMA[4]
    ema20 = HLOCV_EMA[5]
    
    vol_change = (volume[i] - volume[i - 1]) / volume[i - 1]
    momentum = close[i] - close[i - 2]
    volatility = np.std(close[i - 20:i])
    ema_slope = ema20[i] - ema20[i - 1]
    percent_change = (close[i] - close[i - 1]) / close[i - 1]

    features = np.array([[volatility, momentum, vol_change, ema_slope, percent_change]])

    # Predict probability of a successful trade (1 = take profit, 0 = stopped out)
    prob = model.predict_proba(features)[0][1]  # Probability of class 1

    return prob > 0.5  
   