#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:37:50 2025

@author: maurihall
"""
import os
import pickle
import warnings
from time import sleep
from datetime import time, timedelta

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

from ib_insync import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier, XGBRegressor

from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from ib_insync import *
import nest_asyncio
import pandas as pd
nest_asyncio.apply()

from .commissions import calculate_commissions
from .slippage import simulate_slippage
from .signals import candle_conditions
from .feature_engineering import extract_trade_features
from .signals import conditions
from .signals import candle_conditions
from .signals import should_take_trade
from collections import namedtuple
from .signals import conditions_vectorized, should_take_trade_vectorized


def optimized_ml_filtered_backtest(df,
                                    model=None,
                                    risk_reward=1.27,
                                    tick_size=0.01,
                                    buffer=5,
                                    drpt=160,
                                    model_type='rf',
                                    overnight=True,
                                    random_state=None,
                                    verbose=False,
                                    dollar_risk=200,
                                    asset_class='crypto'):
    if verbose:
        print(f"overnight is {overnight}")

    df = df.copy()
    df['ema20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df.dropna(inplace=True)

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    open_ = df['Open'].values
    ema20 = df['ema20'].values
    volume = df['Volume'].values
    index = df.index.to_numpy()
    HLOCV_EMA = (high, low, open_, close, volume, ema20)

    if model is None and model_type in ('xgb', 'rf'):
        raise ValueError("You must pass a trained model for ML filtering.")

    slippage1 = simulate_slippage(len(df))
    slippage2 = simulate_slippage(len(df))

    Trade = namedtuple('Trade', ['time', 'entry', 'stop', 'tp', 'risk_per_unit', 'position_size',
                                 'profit', 'commission', 'net_profit', 'status', 'exit', 'exit_time'])

    trades = []
    in_trade = False

    for i in range(20, len(df) - (buffer + 2)):
        if not in_trade:
            if conditions_vectorized(i, HLOCV_EMA):  # you must implement a vectorized conditions
                take_the_trade = should_take_trade_vectorized(i, HLOCV_EMA, model) if model_type in ('rf', 'xgb') else True

                if take_the_trade:
                    trigger_price = high[i] + tick_size
                    for j in range(1, buffer + 1):
                        if high[i + j] >= trigger_price + slippage1[i]:
                            entry_index = i + j
                            entry_price = trigger_price + slippage1[i]
                            stop_price = np.min(low[i - 12:i])
                            risk_per_unit = entry_price - stop_price
                            take_profit = entry_price + risk_per_unit * risk_reward

                            position_size, commission = calculate_commissions(entry_price, stop_price,
                                                                               dollar_risk, 0.002, asset_class)

                            trades.append(Trade(index[entry_index], entry_price, stop_price, take_profit,
                                                risk_per_unit, position_size, 'open', 'open', 'open',
                                                'open', None, None))

                            in_trade = True
                            trade_open_index = entry_index
                            break
        else:
            for j in range(trade_open_index + 1, len(df)):
                current_time = index[j]
                current_date = current_time.date()
                entry_time = index[trade_open_index]
                entry_date = entry_time.date()
                exit_cutoff = entry_time + timedelta(hours=24)
                close_price = close[j] + slippage2[j]

                position_size = trades[-1].position_size
                commission = calculate_commissions(entry_price, stop_price, dollar_risk, 0.002, asset_class)[1]

                if j + 1 >= len(df):
                    profit = (entry_price - close_price) * position_size
                    trades[-1] = trades[-1]._replace(status='expired', exit=close_price,
                                                     exit_time=current_time, profit=profit,
                                                     commission=commission,
                                                     net_profit=profit - commission)
                    in_trade = False
                    break

                if low[j] <= trades[-1].stop:
                    exit_price = trades[-1].stop + slippage2[j]
                    profit = (exit_price - entry_price) * position_size
                    trades[-1] = trades[-1]._replace(status='stopped_out', exit=exit_price,
                                                     exit_time=current_time, profit=profit,
                                                     commission=commission,
                                                     net_profit=profit - commission)
                    in_trade = False
                    break

                if high[j] >= trades[-1].tp:
                    exit_price = trades[-1].tp + slippage2[j]
                    profit = (exit_price - entry_price) * position_size
                    trades[-1] = trades[-1]._replace(status='take_profit', exit=exit_price,
                                                     exit_time=current_time, profit=profit,
                                                     commission=commission,
                                                     net_profit=profit - commission)
                    in_trade = False
                    break

                if current_time >= exit_cutoff or (current_time - entry_time).total_seconds() > 86400:
                    exit_price = open_[j + 1]
                    next_time = index[j + 1]
                    profit = (exit_price - entry_price) * position_size
                    trades[-1] = trades[-1]._replace(status='timed_out', exit=exit_price,
                                                     exit_time=next_time, profit=profit,
                                                     commission=commission,
                                                     net_profit=profit - commission)
                    in_trade = False
                    break

                if overnight and current_date != entry_date:
                    exit_price = open_[j + 1]
                    next_time = index[j + 1]
                    profit = (exit_price - entry_price) * position_size
                    trades[-1] = trades[-1]._replace(status='overnight_exit', exit=exit_price,
                                                     exit_time=next_time, profit=profit,
                                                     commission=commission,
                                                     net_profit=profit - commission)
                    in_trade = False
                    break

    return pd.DataFrame(trades)

def ml_filtered_backtest(df, 
                         model = None,
                         risk_reward=1.27, 
                         tick_size=0.01, 
                         buffer=5, 
                         drpt=160,
                         model_type='rf',
                         overnight=True, 
                         random_state=None, 
                        verbose=False,
                        dollar_risk=200,
                        asset_class= 'crypto'):
    if verbose:
        print(f"overnight is {overnight}")
    df = df.copy()
    df['ema20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df.dropna(inplace=True)
    
    trades = []
    in_trade = False
    
    if model is None and model_type in ('xgb', 'rf'):
        raise ValueError("You must pass a trained model for ML filtering.")
    slippage_list1 = simulate_slippage(n = len(df))
    slippage_list2 = simulate_slippage(n = len(df))
    for i in range(20, len(df) - (buffer + 2)):
        if not in_trade:
            c1 = df.iloc[i - 2]
            c2 = df.iloc[i - 1]
            c3 = df.iloc[i]
            if conditions(c1, c2, c3):
                df_slice = df.iloc[i - 20:i]
                take_the_trade = (
                    should_take_trade(c1, c2, c3, df_slice, model)
                    if model_type in ('rf', 'xgb') else True
                )
                if take_the_trade:
                    signal_bar_high = c3['High']
                    trigger_price = signal_bar_high + tick_size
                    slippage1 = slippage_list1[i] #tick_size * 3
                    slippage2 = slippage_list2[i]
                    for j in range(1, buffer + 1):
                        future_bar = df.iloc[i + j]
                        if future_bar['High'] >= trigger_price + slippage_list1[i]:
                            entry_index = i + j
                            entry_price = trigger_price + slippage_list1[i]
                            
                            stop_price = df.iloc[i - 12:i]['Low'].min()
                            risk_per_unit = entry_price - stop_price
                            take_profit = entry_price + risk_per_unit * risk_reward
                            position_size, commission = calculate_commissions(entry_price=entry_price,
                                                                              stop_price=stop_price,
                                                                              dollar_risk=dollar_risk,
                                                                              fee_rate=0.002, 
                                                                              asset_class=asset_class)

                            
                            trades.append({
                                'time': df.index[entry_index],
                                'entry': entry_price,
                                'stop': stop_price,
                                'tp': take_profit,
                                'risk_per_unit': risk_per_unit,
                                'position_size': position_size,
                                'profit': 'open',
                                'commission': 'open',
                                'net_profit': 'open',
                                'status': 'open'
                            })
                            in_trade = True
                            trade_open_index = entry_index
                            break
        else:
            for j in range(trade_open_index + 1, len(df)):
                bar = df.iloc[j]
                current_time = df.index[j]
                current_date = current_time.date()
                entry_time = df.index[trade_open_index]
                exit_cutoff = entry_time + timedelta(hours=24)
                entry_date = entry_time.date()  
                low, high = bar['Low'], bar['High']
                close_price = bar['Close'] + slippage_list2[j]
                open_price = bar['Open']
                position_size, commission = calculate_commissions(entry_price = entry_price, 
                                                        stop_price = stop_price, 
                                                        dollar_risk = dollar_risk, 
                                                        fee_rate = 0.002, 
                                                        asset_class = asset_class)

                # Prevent index error
                if j + 1 >= len(df):
                    
                    trades[-1]['status'] = 'expired'
                    trades[-1]['exit'] = close_price 
                    trades[-1]['exit_time'] = current_time
                    trades[-1]['profit'] = (entry_price  - close_price) * position_size
                    trades[-1]['commission'] = commission
                    trades[-1]['position_size'] = position_size
                    trades[-1]['net_profit'] = (entry_price  - close_price) * position_size - commission
                    
                    in_trade = False
                    break

                # Stop loss hit
                if low <= trades[-1]['stop']:
                    profit_before_comm = (trades[-1]['stop'] + slippage_list2[j] - entry_price ) * position_size
                    trades[-1]['status'] = 'stopped_out'
                    trades[-1]['exit'] = trades[-1]['stop'] + slippage_list2[j]
                    trades[-1]['exit_time'] = current_time
                    trades[-1]['profit'] = profit_before_comm
                    trades[-1]['commission'] = commission
                    trades[-1]['position_size'] = position_size
                    trades[-1]['net_profit'] = profit_before_comm - commission
                    in_trade = False
                    break

                # Take profit hit
                elif high >= trades[-1]['tp']:
                    profit_before_comm = (trades[-1]['tp'] + slippage_list2[j] - entry_price ) * position_size
                    trades[-1]['status'] = 'take_profit'
                    trades[-1]['exit'] = trades[-1]['tp'] + slippage_list2[j]
                    trades[-1]['exit_time'] = current_time
                    trades[-1]['profit'] = profit_before_comm
                    trades[-1]['commission'] = commission
                    trades[-1]['position_size'] = position_size
                    trades[-1]['net_profit'] = profit_before_comm - commission
                    in_trade = False
                    break
                
                if current_time >= exit_cutoff:
                   
                    next_open = df.iloc[j + 1]['Open']
                    next_time = df.index[j + 1]
                    profit_before_comm = (next_open - entry_price) * position_size
                    trades[-1]['status'] = 'timed_out'
                    trades[-1]['exit'] = next_open
                    trades[-1]['exit_time'] = next_time
                    trades[-1]['profit'] = profit_before_comm
                    trades[-1]['commission'] = commission
                    trades[-1]['position_size'] = position_size
                    trades[-1]['net_profit'] = profit_before_comm - commission
                    in_trade = False
                    break
                max_hold_minutes = 1440  # e.g., 24 hours
                if (current_time - entry_time).total_seconds() > max_hold_minutes * 60:
                    
                    next_open = df.iloc[j + 1]['Open']
                    next_time = df.index[j + 1]
                    profit_before_comm = (next_open - entry_price) * position_size
                    trades[-1]['status'] = 'timed_out_24'
                    trades[-1]['exit'] = next_open
                    trades[-1]['exit_time'] = next_time
                    trades[-1]['profit'] = profit_before_comm
                    trades[-1]['commission'] = commission
                    trades[-1]['position_size'] = position_size
                    trades[-1]['net_profit'] = profit_before_comm - commission
                    in_trade = False
                    break
                 #Overnight exit
                if overnight and current_date != entry_date:
                    next_open = df.iloc[j + 1]['Open']
                    next_time = df.index[j + 1]
                    profit_before_comm = (next_open - entry_price) * position_size
                    trades[-1]['status'] = 'overnight_exit'
                    trades[-1]['exit'] = next_open
                    trades[-1]['exit_time'] = next_time
                    trades[-1]['profit'] = (next_open - entry_price) * position_size
                    trades[-1]['commission'] = commission
                    trades[-1]['position_size'] = position_size
                    trades[-1]['net_profit'] = profit_before_comm - commission
                    in_trade = False
                    break

                # Forced intraday close
                #if not overnight and current_date == entry_date and current_time.time() >= time(15, 55):
                   # trades[-1]['status'] = 'forced_close'
                   # trades[-1]['exit'] = close_price
                   # trades[-1]['exit_time'] = current_time
                   # trades[-1]['profit'] = (entry_price - close_price) * position_size
                   # in_trade = False
                   # break

    return pd.DataFrame(trades)

def test_trades(df, ticker, random_state=None, overnight=True, asset_class = 'equity'):
    results = []
    print("This is test_trades"*4)
    # Step 1: Create features from full df
    features_df = collect_trade_data(df)

    # Step 2: Time-based split (70/30)
    train_size = int(len(features_df) * 0.7)
    train_df = features_df.iloc[:train_size]
    test_df = features_df.iloc[train_size:]

    X_train = train_df.drop(columns='outcome')
    y_train = train_df['outcome']

    # Step 3: Train the models
    models = {
        'xgb': model_select(X_train, y_train, model_type='xgb'),
        'rf': model_select(X_train, y_train, model_type='rf'),
        'no_ml': None
    }

    for m_type in ('no_ml', 'xgb', 'rf'):
        model = models[m_type]

        # Correctly handle model_type for no_ml
        mt = m_type if m_type in ('xgb', 'rf') else None

        try:
            ml_df = ml_filtered_backtest(
                df,
                model=model,
                model_type=mt,
                random_state=random_state,
                overnight=overnight,
                asset_class = asset_class
            )
        except ValueError as e:
            print(f"Skipping {m_type} due to error: {e}")
            continue

        print(f"{m_type} backtest completed with shape: {ml_df.shape}")

        if ml_df.empty or 'status' not in ml_df.columns:
            print(f"No valid trades for {m_type}")
            continue

        total_profit = ml_df.profit.sum()
        net_profit  = ml_df.net_profit.sum()
        total_commission = ml_df.commission.sum()
        print(f"{ticker} | {m_type} | Total Commission: {round( total_commission)} | Net Profit : {round(net_profit)}")

        results.append({
            'ticker': ticker,
            'model_type': m_type,
            'total_profit': total_profit,
            'net_profit': net_profit,
            'total_commission' : total_commission
        })

    results_df = pd.DataFrame(results)
    return results_df, ml_df



