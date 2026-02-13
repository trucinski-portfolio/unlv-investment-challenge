#!/usr/bin/env python3
"""
ASTRYX INVESTING - Feature Engineering & Time-Series Processor
Reads daily CSV snapshots, merges into a master DataFrame, and computes
30-day delta (velocity) features for trend detection.

Usage:
    python processor.py                    # Process all snapshots
    python processor.py --date 2026-02-10  # Process up to a specific date
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_snapshots')

# Fields to compute 30-day deltas on (the "velocity" features)
DELTA_FIELDS = [
    'forwardPE',          # Value Compression detection
    'revenueGrowth',      # Growth Acceleration detection
    'operatingMargins',   # Efficiency Gains detection
    'returnOnEquity',     # ROE trend
    'debtToEquity',       # Leverage trend
    'currentPrice',       # Price momentum
]

# Sector averages used for NaN imputation
# These are approximate S&P 500 sector medians and serve as fallbacks
SECTOR_DEFAULTS = {
    'Technology':        {'trailingPE': 28, 'forwardPE': 25, 'pegRatio': 1.8, 'revenueGrowth': 0.10, 'returnOnEquity': 0.25, 'debtToEquity': 60, 'operatingMargins': 0.25, 'currentRatio': 1.8},
    'Financial Services': {'trailingPE': 14, 'forwardPE': 12, 'pegRatio': 1.2, 'revenueGrowth': 0.05, 'returnOnEquity': 0.12, 'debtToEquity': 150, 'operatingMargins': 0.30, 'currentRatio': 1.0},
    'Healthcare':        {'trailingPE': 22, 'forwardPE': 18, 'pegRatio': 1.5, 'revenueGrowth': 0.08, 'returnOnEquity': 0.20, 'debtToEquity': 80, 'operatingMargins': 0.20, 'currentRatio': 1.5},
    'Consumer Cyclical': {'trailingPE': 20, 'forwardPE': 18, 'pegRatio': 1.4, 'revenueGrowth': 0.06, 'returnOnEquity': 0.18, 'debtToEquity': 90, 'operatingMargins': 0.12, 'currentRatio': 1.3},
    'Industrials':       {'trailingPE': 20, 'forwardPE': 17, 'pegRatio': 1.5, 'revenueGrowth': 0.05, 'returnOnEquity': 0.18, 'debtToEquity': 80, 'operatingMargins': 0.15, 'currentRatio': 1.4},
    'Energy':            {'trailingPE': 12, 'forwardPE': 10, 'pegRatio': 1.0, 'revenueGrowth': 0.03, 'returnOnEquity': 0.15, 'debtToEquity': 50, 'operatingMargins': 0.15, 'currentRatio': 1.2},
    'Communication Services': {'trailingPE': 18, 'forwardPE': 16, 'pegRatio': 1.3, 'revenueGrowth': 0.07, 'returnOnEquity': 0.15, 'debtToEquity': 80, 'operatingMargins': 0.20, 'currentRatio': 1.3},
    'Consumer Defensive': {'trailingPE': 22, 'forwardPE': 20, 'pegRatio': 2.0, 'revenueGrowth': 0.03, 'returnOnEquity': 0.20, 'debtToEquity': 100, 'operatingMargins': 0.12, 'currentRatio': 1.0},
    'Utilities':         {'trailingPE': 18, 'forwardPE': 16, 'pegRatio': 2.5, 'revenueGrowth': 0.02, 'returnOnEquity': 0.10, 'debtToEquity': 130, 'operatingMargins': 0.20, 'currentRatio': 0.8},
    'Real Estate':       {'trailingPE': 35, 'forwardPE': 30, 'pegRatio': 3.0, 'revenueGrowth': 0.04, 'returnOnEquity': 0.08, 'debtToEquity': 100, 'operatingMargins': 0.30, 'currentRatio': 0.5},
    'Basic Materials':   {'trailingPE': 15, 'forwardPE': 13, 'pegRatio': 1.2, 'revenueGrowth': 0.04, 'returnOnEquity': 0.14, 'debtToEquity': 60, 'operatingMargins': 0.15, 'currentRatio': 1.5},
}


def list_snapshot_files(data_dir: str = None) -> list:
    """Return sorted list of (date_str, filepath) tuples from raw_snapshots/."""
    if data_dir is None:
        data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        return []

    files = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('_fundamentals.csv'):
            date_str = f.replace('_fundamentals.csv', '')
            filepath = os.path.join(data_dir, f)
            files.append((date_str, filepath))
    return files


def master_merge(data_dir: str = None, up_to_date: str = None) -> pd.DataFrame:
    """
    Read all CSVs in data/raw_snapshots/ and create a single long-format DataFrame.
    Each row = (ticker, date, metric1, metric2, ...).

    Args:
        data_dir: Path to raw_snapshots directory
        up_to_date: Only include snapshots up to this date (YYYY-MM-DD)

    Returns:
        Long-format DataFrame with 'snapshot_date' and 'ticker' columns
    """
    files = list_snapshot_files(data_dir)
    if not files:
        print("  No snapshot files found!")
        return pd.DataFrame()

    if up_to_date:
        files = [(d, f) for d, f in files if d <= up_to_date]

    frames = []
    for date_str, filepath in files:
        try:
            df = pd.read_csv(filepath)
            df['snapshot_date'] = date_str
            frames.append(df)
        except Exception as e:
            print(f"  Warning: skipping {filepath}: {e}")

    if not frames:
        return pd.DataFrame()

    master = pd.concat(frames, ignore_index=True)
    master['snapshot_date'] = pd.to_datetime(master['snapshot_date'])
    master = master.sort_values(['ticker', 'snapshot_date']).reset_index(drop=True)

    print(f"  Master merge: {len(master)} rows, {master['snapshot_date'].nunique()} dates, "
          f"{master['ticker'].nunique()} tickers")

    return master


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing data (NaNs):
    1. Forward-fill within each ticker's time series
    2. Fill remaining NaNs with sector averages
    3. Fill any still-missing with global medians
    """
    if df.empty:
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude snapshot_date from numeric processing
    numeric_cols = [c for c in numeric_cols if c != 'snapshot_date']

    # Step 1: Forward-fill within each ticker
    df = df.sort_values(['ticker', 'snapshot_date'])
    for col in numeric_cols:
        df[col] = df.groupby('ticker')[col].ffill()

    # Step 2: Fill with sector averages
    if 'sector' in df.columns:
        for sector, defaults in SECTOR_DEFAULTS.items():
            mask = df['sector'] == sector
            for field, default_val in defaults.items():
                if field in df.columns:
                    df.loc[mask, field] = df.loc[mask, field].fillna(default_val)

    # Step 3: Fill remaining with global medians
    for col in numeric_cols:
        median_val = df[col].median()
        if pd.notna(median_val):
            df[col] = df[col].fillna(median_val)

    return df


def compute_deltas(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """
    Compute 30-day percentage change ("velocity") for key metrics.

    For each ticker, calculates:
        delta_<field> = (current - value_30d_ago) / |value_30d_ago| * 100

    This detects:
        - Value Compression: forwardPE delta < 0 (P/E shrinking = getting cheaper)
        - Growth Acceleration: revenueGrowth delta > 0 (growth rate increasing)
        - Efficiency Gains: operatingMargins delta > 0 (margins expanding)
    """
    if df.empty or 'snapshot_date' not in df.columns:
        return df

    df = df.sort_values(['ticker', 'snapshot_date']).reset_index(drop=True)

    # Find the lookback window in terms of rows per ticker
    dates = sorted(df['snapshot_date'].unique())
    if len(dates) < 2:
        print(f"  Only {len(dates)} date(s) available, skipping delta calculations")
        for field in DELTA_FIELDS:
            df[f'delta_{field}'] = np.nan
        return df

    # For each ticker, compute deltas
    for field in DELTA_FIELDS:
        if field not in df.columns:
            df[f'delta_{field}'] = np.nan
            continue

        delta_col = f'delta_{field}'
        df[delta_col] = np.nan

        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df.loc[mask].copy()

            if len(ticker_data) < 2:
                continue

            # Find the row closest to `lookback_days` ago
            latest_date = ticker_data['snapshot_date'].max()
            target_date = latest_date - pd.Timedelta(days=lookback_days)

            # Get the closest available date
            available_dates = ticker_data['snapshot_date'].sort_values()
            past_dates = available_dates[available_dates <= target_date]

            if past_dates.empty:
                # Use earliest available date
                past_date = available_dates.iloc[0]
            else:
                past_date = past_dates.iloc[-1]

            current_val = ticker_data.loc[
                ticker_data['snapshot_date'] == latest_date, field
            ].iloc[0]

            past_val = ticker_data.loc[
                ticker_data['snapshot_date'] == past_date, field
            ].iloc[0]

            if pd.notna(current_val) and pd.notna(past_val) and abs(past_val) > 0.001:
                pct_change = ((current_val - past_val) / abs(past_val)) * 100
                df.loc[mask & (df['snapshot_date'] == latest_date), delta_col] = pct_change

    return df


def get_latest_with_trends(data_dir: str = None, up_to_date: str = None) -> pd.DataFrame:
    """
    Full processing pipeline: merge -> impute -> compute deltas -> return latest snapshot.

    Returns a DataFrame with one row per ticker containing:
    - All 24 fundamental fields
    - 30-day delta fields (delta_forwardPE, delta_revenueGrowth, etc.)
    """
    print("\n=== Processing Pipeline ===")

    # Step 1: Master merge
    master = master_merge(data_dir, up_to_date)
    if master.empty:
        return master

    # Step 2: Impute missing data
    print("  Imputing missing data...")
    master = impute_missing(master)

    # Step 3: Compute 30-day deltas
    print("  Computing 30-day deltas...")
    master = compute_deltas(master)

    # Step 4: Extract latest date only
    latest_date = master['snapshot_date'].max()
    latest = master[master['snapshot_date'] == latest_date].copy()
    latest = latest.reset_index(drop=True)

    print(f"  Latest snapshot: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  Tickers with data: {len(latest)}")

    # Report on delta availability
    delta_cols = [c for c in latest.columns if c.startswith('delta_')]
    for col in delta_cols:
        valid = latest[col].notna().sum()
        print(f"    {col}: {valid}/{len(latest)} computed")

    return latest


def main():
    parser = argparse.ArgumentParser(description='Process fundamental snapshots')
    parser.add_argument('--date', '-d', default=None,
                        help='Process up to this date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', default=None,
                        help='Path to raw_snapshots directory')
    args = parser.parse_args()

    df = get_latest_with_trends(data_dir=args.data_dir, up_to_date=args.date)

    if not df.empty:
        print(f"\n  Result: {len(df)} tickers x {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
