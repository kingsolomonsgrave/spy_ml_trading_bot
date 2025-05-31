#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 13:19:39 2025

@author: maurihall
"""

from spy_bot.data_loader import get_yf_data
from spy_bot.evaluator import test_trades

# Load recent SPY data from Yahoo Finance
df = get_yf_data("SPY", period="30d", interval="5m")

# Run test trades with random forest and XGBoost
results, trades = test_trades(df, ticker="SPY")

# Display output
print("\nResults Summary:")
print(results)

print("\nSample Trades:")
print(trades.head())
