#!/usr/bin/env python3
"""
ASTRYX INVESTING - Daily Fundamental Data Fetcher
Fetches 24 fundamental metrics for all S&P 500 stocks and saves as dated CSV.

Usage:
    python3 data_fetcher.py                  # Fetch today's snapshot
    python3 data_fetcher.py --date 2026-02-10  # Fetch for a specific date
"""

import os
import sys
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_service import DataService

# The 24 fundamental "biomarkers" we track
FUNDAMENTAL_FIELDS = [
    # Valuation (5)
    'trailingPE', 'forwardPE', 'pegRatio',
    'priceToSalesTrailing12Months', 'enterpriseToEbitda',
    # Growth (3)
    'revenueGrowth', 'earningsGrowth', 'earningsQuarterlyGrowth',
    # Profitability (5)
    'returnOnEquity', 'profitMargins', 'freeCashflow',
    'operatingMargins', 'grossMargins',
    # Health (4)
    'debtToEquity', 'currentRatio', 'shortPercentOfFloat', 'shortRatio',
    # Identity/Context (7)
    'marketCap', 'beta', 'averageVolume', 'dividendYield',
    'sector', 'industry', 'currentPrice',
]

# Output directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_snapshots')


def fetch_ticker_fundamentals(ticker: str) -> dict:
    """Fetch fundamental data for a single ticker via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        row = {'ticker': ticker}
        for field in FUNDAMENTAL_FIELDS:
            row[field] = info.get(field)
        row['longName'] = info.get('longName', info.get('shortName', ticker))
        return row
    except Exception as e:
        row = {'ticker': ticker, 'longName': ticker}
        for field in FUNDAMENTAL_FIELDS:
            row[field] = None
        row['_error'] = str(e)
        return row


def fetch_daily_snapshot(tickers: list, max_workers: int = 10,
                         show_progress: bool = True) -> pd.DataFrame:
    """
    Fetch fundamentals for all tickers in parallel.
    Returns a DataFrame with one row per ticker and 24+ columns.
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_ticker_fundamentals, t): t for t in tickers}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(tickers), desc="Fetching fundamentals")

        for future in iterator:
            try:
                results.append(future.result())
            except Exception:
                ticker = futures[future]
                results.append({'ticker': ticker, '_error': 'future_failed'})

    df = pd.DataFrame(results)

    # Reorder columns: ticker first, then sorted fields
    cols = ['ticker', 'longName'] + FUNDAMENTAL_FIELDS
    cols = [c for c in cols if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]

    return df


def save_snapshot(df: pd.DataFrame, target_date: str = None) -> str:
    """
    Save DataFrame as dated CSV: data/raw_snapshots/YYYY-MM-DD_fundamentals.csv
    Returns the output path.
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')

    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{target_date}_fundamentals.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    return filepath


def run_fetch(target_date: str = None, top_n: int = None) -> str:
    """
    Full pipeline: get S&P 500 tickers, fetch fundamentals, save CSV.
    Returns path to saved file.
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n=== ASTRYX Daily Fundamental Fetch: {target_date} ===\n")

    # Get ticker list
    tickers = DataService.get_sp500_tickers(top_n)
    print(f"  Tickers: {len(tickers)}")

    # Fetch
    df = fetch_daily_snapshot(tickers)
    valid = df['currentPrice'].notna().sum()
    print(f"  Valid data: {valid}/{len(df)} tickers")

    # Drop error column before saving
    if '_error' in df.columns:
        errors = df['_error'].notna().sum()
        if errors > 0:
            print(f"  Errors: {errors}")
        df = df.drop(columns=['_error'])

    # Save
    filepath = save_snapshot(df, target_date)
    print(f"  Saved: {filepath}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows: {len(df)}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description='Fetch daily S&P 500 fundamentals')
    parser.add_argument('--date', '-d', default=None,
                        help='Target date (YYYY-MM-DD). Default: today')
    parser.add_argument('--top', '-n', type=int, default=None,
                        help='Only fetch top N tickers by market cap')
    args = parser.parse_args()

    run_fetch(target_date=args.date, top_n=args.top)


if __name__ == "__main__":
    main()
