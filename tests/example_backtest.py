#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:38:50 2025

@author: maurihall
"""

from spy_bot.data_loader import get_yf_data
from spy_bot.backtester import test_trades

df = get_yf_data("SPY")
results_df, ml_df_ = test_trades(df, symbol = 'SPY', random_state=None, asset_class = 'equity')
