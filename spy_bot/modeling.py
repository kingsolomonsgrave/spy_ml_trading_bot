#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:37:33 2025

@author: maurihall
"""

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def model_select(X_train, y_train, model_type="rf", ne=200, random_state=42):
    unique_y = set(y_train)
    if unique_y != {0, 1}:
        raise ValueError(f"Invalid target classes for classification: {unique_y}")
    if model_type == 'xgb':
        xgb_model = XGBClassifier(
            n_estimators=ne,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model

    elif model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=ne, random_state=random_state)
        rf_model.fit(X_train, y_train)
        return rf_model



