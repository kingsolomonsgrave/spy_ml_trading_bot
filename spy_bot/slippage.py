#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:38:30 2025

@author: maurihall
"""

import numpy as np

def simulate_slippage(n, order_size=1, USD_liquidity=50_000_000, base_slip = -0.0002):
        L = []
        
        for i in range(n):
            impact = (order_size / USD_liquidity) ** 0.7
            noise = np.random.normal(0, 0.0001)
            slip = base_slip * impact + noise
            slip = np.clip(slip, -0.01, 0.002)
            L.append(slip)
        return L