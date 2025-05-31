#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:38:15 2025

@author: maurihall
"""

def calculate_commissions(entry_price, stop_price, dollar_risk, fee_rate=0.004, asset_class='equity'): 
    if asset_class not in ('crypto', 'equity', 'FX'):
        raise ValueError(f"Unsupported asset_class: {asset_class}. Must be 'crypto', 'equity', or 'FX'.")

    risk_per_unit = entry_price - stop_price
    if risk_per_unit == 0:
        raise ValueError("entry_price and stop_price must not be equal (division by zero).")

    position_size = round(dollar_risk / risk_per_unit, 4)
    trade_value = position_size * entry_price

    if asset_class == "crypto":
        commission = trade_value * fee_rate

    elif asset_class == "equity":
        fee_per_share = 0.005
        extra_fees_per_share = 0.0030
        raw_fee = position_size * (fee_per_share)
        commission = max(1, min(raw_fee, trade_value * 0.01))


    elif asset_class == "FX":
        raise NotImplementedError("FX commissions not yet supported.")

    return position_size, commission