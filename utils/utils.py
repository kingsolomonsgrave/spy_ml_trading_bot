#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:48:42 2025

@author: maurihall
"""

def total_bars_to_time(total_bars, df):
    total_minutes = total_bars*5
    total_hours = total_minutes/60
    total_days = total_hours/25
    n_trades = len(df)
    print(n_trades/total_days)
    print(n_trades)
    print(total_days)