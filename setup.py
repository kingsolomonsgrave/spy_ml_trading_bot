#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 12:41:10 2025

@author: maurihall
"""

from setuptools import setup, find_packages

setup(
    name='spy_ml_trading_bot',
    version='0.1.0',
    author='Dr. Mauri K Hall',
    author_email='maurikhall@gmail.com',
    description='A modular machine learning-based trading bot for SPY and crypto assets using EMA signals and model filtering.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kingsolomonsgrave/spy_ml_trading_bot',
    packages=find_packages(include=['spy_bot', 'spy_bot.*']),
    install_requires=[
        'pandas',
        'numpy',
        'yfinance',
        'ta',
        'xgboost',
        'scikit-learn',
        'ib_insync',
        'requests',
        'nest_asyncio'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # or your choice
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
