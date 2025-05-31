#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:35:20 2025

@author: maurihall
"""

# === Imports ===
import os
import pickle
import warnings
from datetime import time, timedelta
from time import sleep

import nest_asyncio
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf
from ib_insync import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from xgboost import XGBClassifier, XGBRegressor

# === Settings ===
warnings.filterwarnings("ignore")
nest_asyncio.apply()

# === Data Loaders ===

def get_yf_data(ticker="SPY", period="60d", interval="5m", add_ema=True, ema_window=20):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns.name = None

    if add_ema and 'Close' in df.columns:
        df[f'ema{ema_window}'] = EMAIndicator(df['Close'], window=ema_window).ema_indicator()

    return df


async def get_ibkr_data(ticker='SPY'):
    ib = IB()
    await ib.connectAsync('127.0.0.1', 7497, clientId=1)
    print("Connected to TWS!")

    contract = Stock(ticker, 'SMART', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='20241101 23:59:59',
        durationStr='60 D',
        barSizeSetting='5 mins',
        whatToShow='TRADES',
        useRTH=True
    )

    df = util.df(bars)
    df.columns = [col.capitalize() for col in df.columns]
    df['ema20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df.index = df.Date
    df.drop(columns=["Date", "Average", "Barcount"], inplace=True)
    df.columns.name = None
    df = df[['Close', 'High', 'Low', 'Open', 'Volume', 'ema20']]
    ib.disconnect()
    return df


def get_binance_ohlcv_paged(symbol="BTCUSDT", interval="5m", total_bars=4000):
    limit = 1000
    bars = []
    url = "https://api.binance.com/api/v3/klines"
    end_time = None

    for _ in range(total_bars // limit):
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = end_time
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        bars.extend(data)
        end_time = data[0][0] - 1
        sleep(0.2)

    df = pd.DataFrame(bars, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.columns = [col.capitalize() for col in df.columns]
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.rename(columns={'Timestamp': 'Date'}, inplace=True)
    df['ema20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df.set_index('Date', inplace=True)
    df.index.name = 'Datetime'
    return df

# === Utilities ===

def total_bars_to_time(total_bars, n_trades):
    total_minutes = total_bars * 5
    total_hours = total_minutes / 60
    total_days = total_hours / 25
    print(n_trades / total_days)
    print(n_trades)
    print(total_days)

# === Signal Logic ===

def conditions(c1, c2, c3, tail1=0.05, tail2=0.01, larger_body=True, df=None):
    c3_body = abs(c3["Open"] - c3["Close"])
    c2_body = abs(c2["Open"] - c2["Close"])

    cond1 = c1['Close'] < c1['ema20']
    cond2 = c2['Close'] > c2['ema20']
    cond3 = c3['Close'] > c3['ema20']
    cond4 = c3['Close'] > c2['High']
    cond5 = ((c2['High'] - c2['Close']) / c2['Close']) < tail1
    cond6 = ((c3['High'] - c3['Close']) / c3['Close']) < tail2
    cond7 = c3["Volume"] > c2["Volume"] if c3["Volume"] and c2["Volume"] else False
    cond8 = c3_body > c2_body if larger_body else True

    return all([cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8])
