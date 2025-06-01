#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:37:01 2025

@author: maurihall
"""
import numpy as np
import pandas as pd
from .signals import conditions_vectorized



def collect_trade_data_fast(df, risk_reward=1.27, tick_size=0.01, buffer=5, drpt=160):
    features = []
    df = df.copy()
    df.dropna(inplace=True)
    trades = []
    in_trade = False
    n = len(df)
    
    #Numpy Arrays
    highs = df['High'].values
    lows = df['Low'].values
    opens = df['Open'].values
    closes = df['Close'].values
    volumes = df['Volume'].values
    ema20 = df['ema20'].values
    HLOCV_EMA = [highs, lows, opens, closes, volumes, ema20]
    
    rolling_low_12 = pd.Series(lows).rolling(12).min().values
    rolling_std_20 = pd.Series(highs).rolling(20).std().values

    for i in range(20, n - buffer - 2):
        c1_idx = i - 2
        c2_idx = i - 1
        c3_idx = i

        # Use array indexing instead of iloc
        if not (np.isnan(ema20[c1_idx]) or np.isnan(ema20[c2_idx]) or np.isnan(ema20[c3_idx])):
            
            if conditions_vectorized(i, HLOCV_EMA):
                trigger_price = highs[c3_idx] + tick_size
                entry_slippage = tick_size * 2

                for j in range(1, buffer + 1):
                    fut_idx = i + j
                    if highs[fut_idx] >= trigger_price + entry_slippage:
                        entry_idx = fut_idx
                        entry_price = trigger_price + entry_slippage
                        stop_price = rolling_low_12[i]
                        risk_per_unit = entry_price - stop_price
                        take_profit = entry_price + risk_per_unit * risk_reward

                        # Compute features
                        vol_change = (volumes[c3_idx] - volumes[c2_idx]) / volumes[c2_idx]
                        momentum = closes[c3_idx] - closes[c1_idx]
                        volatility = rolling_std_20[i]
                        ema_slope = ema20[c3_idx] - ema20[c2_idx]
                        percent_change = (closes[c3_idx] - closes[c2_idx]) / closes[c2_idx]

                        # Simulate outcome (stop or TP)
                        outcome = 0
                        for k in range(entry_idx + 1, n):
                            if lows[k] <= stop_price:
                                break
                            elif highs[k] >= take_profit:
                                outcome = 1
                                break

                        features.append({
                            'volatility': volatility,
                            'momentum': momentum,
                            'vol_change': vol_change,
                            'ema_slope': ema_slope,
                            'percent_change': percent_change,
                            'outcome': outcome
                        })

                        break  # exit buffer loop

    return pd.DataFrame(features)






def extract_trade_features(df_slice, c1, c2, c3):
    return {
        'volatility': df_slice['High'].std(),
        'momentum': c3['Close'] - c1['Close'],
        'vol_change': (c3['Volume'] - c2['Volume']) / c2['Volume'],
        'ema_slope': c3['ema20'] - c2['ema20'],
        'percent_change': (c3['Close'] - c2['Close']) / c2['Close']
    }



import pandas as pd

from .signals import conditions

def collect_trade_data(df, risk_reward=1.27, tick_size=0.01, buffer=5, drpt=160):
    """
    Scans the OHLCV DataFrame for trades and extracts labeled features.

    Parameters:
    - df: DataFrame with price data and 'ema20'
    - risk_reward: Risk-reward multiplier
    - tick_size: Minimum price movement
    - buffer: How many bars ahead to wait for confirmation
    - drpt: Unused currently (reserved for future use)

    Returns:
    - DataFrame with features and binary outcome (0/1)
    """
    feature_list = []
    df = df.copy()
    df.dropna(inplace=True)
    in_trade = False

    for i in range(20, len(df) - (buffer + 2)):
        if not in_trade:
            c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
            if conditions(c1, c2, c3, df=df):
                signal_bar_high = c3['High']
                trigger_price = signal_bar_high + tick_size
                slippage = tick_size * 3

                for j in range(1, buffer + 1):
                    future_bar = df.iloc[i + j]
                    if future_bar['High'] >= trigger_price + slippage:
                        entry_index = i + j
                        entry_price = trigger_price + slippage
                        stop_price = df.iloc[i - 12:i]['Low'].min()
                        risk_per_unit = entry_price - stop_price
                        take_profit = entry_price + risk_per_unit * risk_reward

                        # Feature engineering
                        features = {
                            'volatility': df.iloc[i - 20:i]['High'].std(),
                            'momentum': c3['Close'] - c1['Close'],
                            'vol_change': (c3['Volume'] - c2['Volume']) / c2['Volume'],
                            'ema_slope': c3['ema20'] - c2['ema20'],
                            'percent_change': (c3['Close'] - c2['Close']) / c2['Close'],
                            'outcome': 0  # default to failure
                        }

                        for k in range(entry_index + 1, len(df)):
                            bar = df.iloc[k]
                            if bar['Low'] <= stop_price:
                                break  # stopped out
                            elif bar['High'] >= take_profit:
                                features['outcome'] = 1
                                break  # target hit

                        feature_list.append(features)
                        break  # move to next scan window

    return pd.DataFrame(feature_list)

