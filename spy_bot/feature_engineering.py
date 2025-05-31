#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:37:01 2025

@author: maurihall
"""

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

