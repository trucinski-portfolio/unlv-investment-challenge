#!/usr/bin/env python3
"""
ASTRYX INVESTING - Historical Backfiller
One-time script to backfill daily fundamental CSVs from 2026-01-01 to today.

Since yfinance fundamentals are point-in-time (current values only),
this script fetches today's fundamentals once and replicates them across
historical trading dates. For price data, it uses actual historical prices.

Usage:
    python historical_backfiller.py                        # Backfill from 2026-01-01
    python historical_backfiller.py --start 2026-01-15     # Custom start date
    python historical_backfiller.py --skip-existing        # Skip dates already fetched
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_fetcher import (
    fetch_daily_snapshot, save_snapshot, FUNDAMENTAL_FIELDS, DATA_DIR,
)
from data_service import DataService


def get_trading_dates(start_date: str, end_date: str = None) -> list:
    """
    Get actual trading dates between start and end using SPY history.
    This filters out weekends and market holidays.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    spy = yf.Ticker('SPY')
    hist = spy.history(start=start_date, end=end_date)

    if hist.empty:
        print("  Warning: No SPY data for date range, using calendar days")
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Mon-Fri
                dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        return dates

    dates = [d.strftime('%Y-%m-%d') for d in hist.index]
    return dates


def get_existing_dates() -> set:
    """Return set of dates that already have CSVs in the snapshots folder."""
    if not os.path.exists(DATA_DIR):
        return set()
    existing = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith('_fundamentals.csv'):
            date_str = f.replace('_fundamentals.csv', '')
            existing.add(date_str)
    return existing


def backfill(start_date: str = '2026-01-01', end_date: str = None,
             skip_existing: bool = True):
    """
    Backfill historical daily CSVs.

    Strategy:
    1. Fetch current fundamentals once (they don't change retroactively).
    2. Fetch historical price data for the full range.
    3. For each trading date, merge the fundamentals with that day's
       closing price, creating a dated CSV.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n=== ASTRYX Historical Backfiller ===")
    print(f"  Range: {start_date} to {end_date}")

    # Step 1: Get trading dates
    trading_dates = get_trading_dates(start_date, end_date)
    print(f"  Trading days in range: {len(trading_dates)}")

    existing = get_existing_dates()
    if skip_existing:
        dates_to_fill = [d for d in trading_dates if d not in existing]
        print(f"  Already fetched: {len(existing)}")
        print(f"  Remaining: {len(dates_to_fill)}")
    else:
        dates_to_fill = trading_dates

    if not dates_to_fill:
        print("  Nothing to backfill!")
        return

    # Step 2: Fetch current fundamentals (one-time batch)
    tickers = DataService.get_sp500_tickers()
    print(f"\n  Fetching fundamentals for {len(tickers)} tickers...")
    fundamentals_df = fetch_daily_snapshot(tickers, max_workers=10)

    # Drop error column
    if '_error' in fundamentals_df.columns:
        fundamentals_df = fundamentals_df.drop(columns=['_error'])

    # Step 3: Fetch historical prices for the full range
    print(f"\n  Fetching historical prices ({start_date} to {end_date})...")
    price_data = yf.download(
        tickers, start=start_date, end=end_date,
        group_by='ticker', threads=True, progress=True,
    )

    # Step 4: For each trading date, create a snapshot
    print(f"\n  Generating daily snapshots...")
    for date_str in tqdm(dates_to_fill, desc="Backfilling"):
        try:
            date_dt = pd.to_datetime(date_str)
            daily_df = fundamentals_df.copy()

            # Override currentPrice with actual historical close
            for i, row in daily_df.iterrows():
                ticker = row['ticker']
                try:
                    if len(tickers) == 1:
                        close = price_data.loc[date_dt, 'Close']
                    else:
                        close = price_data.loc[date_dt, (ticker, 'Close')]
                    if pd.notna(close):
                        daily_df.at[i, 'currentPrice'] = float(close)
                except (KeyError, TypeError):
                    pass  # Keep original value or NaN

            save_snapshot(daily_df, date_str)

        except Exception as e:
            print(f"\n  Error on {date_str}: {e}")
            continue

    final_count = len(get_existing_dates())
    print(f"\n  Backfill complete! Total snapshots: {final_count}")


def main():
    parser = argparse.ArgumentParser(description='Backfill historical fundamental CSVs')
    parser.add_argument('--start', '-s', default='2026-01-01',
                        help='Start date (YYYY-MM-DD). Default: 2026-01-01')
    parser.add_argument('--end', '-e', default=None,
                        help='End date (YYYY-MM-DD). Default: today')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip dates that already have CSVs (default: True)')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetch all dates even if CSVs exist')
    args = parser.parse_args()

    skip = not args.force
    backfill(start_date=args.start, end_date=args.end, skip_existing=skip)


if __name__ == "__main__":
    main()
