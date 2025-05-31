#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:59:12 2025

@author: maurihall

"""
from .modeling import model_select
from .feature_engineering import collect_trade_data
from .backtester import ml_filtered_backtest
import pandas as pd

def test_trades(df, ticker, random_state=None, overnight=True, asset_class = 'equity'):
    results = []

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
