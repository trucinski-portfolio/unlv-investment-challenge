#!/usr/bin/env python3
"""
ASTRYX INVESTING - Data Validator (Sanity Check)
Verifies each daily CSV has ~500 rows and all required columns.

Usage:
    python validator.py                      # Validate all snapshots
    python validator.py --date 2026-02-10    # Validate a specific date
    python validator.py --fix                # Attempt to fix issues
"""

import os
import sys
import argparse
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_fetcher import FUNDAMENTAL_FIELDS, DATA_DIR

# Expected row count range (S&P 500 = ~500-505 stocks)
MIN_ROWS = 480
MAX_ROWS = 520

# Required columns (must be present in every CSV)
REQUIRED_COLUMNS = ['ticker', 'longName'] + FUNDAMENTAL_FIELDS


def validate_file(filepath: str, verbose: bool = True) -> dict:
    """
    Validate a single CSV file.

    Returns dict with:
        - valid: bool
        - filepath: str
        - issues: list of issue strings
        - row_count: int
        - col_count: int
        - missing_cols: list
        - null_heavy_cols: list (cols with >50% NaN)
    """
    result = {
        'valid': True,
        'filepath': filepath,
        'issues': [],
        'row_count': 0,
        'col_count': 0,
        'missing_cols': [],
        'null_heavy_cols': [],
    }

    # Check file exists
    if not os.path.exists(filepath):
        result['valid'] = False
        result['issues'].append(f"File not found: {filepath}")
        return result

    # Read CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        result['valid'] = False
        result['issues'].append(f"Cannot read CSV: {e}")
        return result

    result['row_count'] = len(df)
    result['col_count'] = len(df.columns)

    # Check row count
    if len(df) < MIN_ROWS:
        result['valid'] = False
        result['issues'].append(f"Too few rows: {len(df)} (expected {MIN_ROWS}-{MAX_ROWS})")
    elif len(df) > MAX_ROWS:
        result['issues'].append(f"More rows than expected: {len(df)} (expected {MIN_ROWS}-{MAX_ROWS})")

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        result['valid'] = False
        result['missing_cols'] = missing
        result['issues'].append(f"Missing columns: {missing}")

    # Check for columns with excessive NaN
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            null_pct = df[col].isna().mean()
            if null_pct > 0.50:
                result['null_heavy_cols'].append((col, f"{null_pct*100:.0f}%"))

    if result['null_heavy_cols']:
        result['issues'].append(
            f"High null rate: {', '.join(f'{c}({p})' for c, p in result['null_heavy_cols'])}"
        )

    # Check for duplicate tickers
    if 'ticker' in df.columns:
        dupes = df['ticker'].duplicated().sum()
        if dupes > 0:
            result['issues'].append(f"Duplicate tickers: {dupes}")

    # Check for valid price data
    if 'currentPrice' in df.columns:
        valid_prices = df['currentPrice'].notna().sum()
        price_pct = valid_prices / len(df)
        if price_pct < 0.50:
            result['valid'] = False
            result['issues'].append(f"Only {valid_prices}/{len(df)} ({price_pct*100:.0f}%) have valid prices")
        elif price_pct < 0.80:
            result['issues'].append(f"Partial prices: {valid_prices}/{len(df)} ({price_pct*100:.0f}%)")

    return result


def validate_all(data_dir: str = None, verbose: bool = True) -> list:
    """
    Validate all CSV files in the raw_snapshots directory.
    Returns list of validation results.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if not os.path.exists(data_dir):
        print(f"  Data directory not found: {data_dir}")
        return []

    files = sorted([
        f for f in os.listdir(data_dir) if f.endswith('_fundamentals.csv')
    ])

    if not files:
        print("  No snapshot files found!")
        return []

    print(f"\n=== ASTRYX Data Validator ===")
    print(f"  Directory: {data_dir}")
    print(f"  Files: {len(files)}")
    print(f"  Expected rows: {MIN_ROWS}-{MAX_ROWS}")
    print(f"  Required columns: {len(REQUIRED_COLUMNS)}")
    print()

    results = []
    valid_count = 0
    warning_count = 0

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        result = validate_file(filepath, verbose)
        results.append(result)

        date_str = filename.replace('_fundamentals.csv', '')

        if result['valid'] and not result['issues']:
            valid_count += 1
            if verbose:
                print(f"  OK    {date_str}  ({result['row_count']} rows, {result['col_count']} cols)")
        elif result['valid'] and result['issues']:
            warning_count += 1
            if verbose:
                print(f"  WARN  {date_str}  ({result['row_count']} rows)")
                for issue in result['issues']:
                    print(f"         -> {issue}")
        else:
            if verbose:
                print(f"  FAIL  {date_str}")
                for issue in result['issues']:
                    print(f"         -> {issue}")

    print(f"\n  Summary: {valid_count} OK, {warning_count} warnings, "
          f"{len(results) - valid_count - warning_count} failures out of {len(results)} files")

    return results


def validate_date(date_str: str, data_dir: str = None, verbose: bool = True) -> dict:
    """Validate a specific date's CSV."""
    if data_dir is None:
        data_dir = DATA_DIR

    filename = f"{date_str}_fundamentals.csv"
    filepath = os.path.join(data_dir, filename)

    print(f"\n=== Validating: {date_str} ===")
    result = validate_file(filepath, verbose)

    if result['valid'] and not result['issues']:
        print(f"  Status: OK")
    elif result['valid']:
        print(f"  Status: OK (with warnings)")
    else:
        print(f"  Status: FAILED")

    print(f"  Rows: {result['row_count']}")
    print(f"  Columns: {result['col_count']}")

    if result['missing_cols']:
        print(f"  Missing: {result['missing_cols']}")
    if result['null_heavy_cols']:
        print(f"  High null: {result['null_heavy_cols']}")
    for issue in result['issues']:
        print(f"  Issue: {issue}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Validate daily CSV snapshots')
    parser.add_argument('--date', '-d', default=None,
                        help='Validate a specific date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', default=None,
                        help='Path to raw_snapshots directory')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Only show failures')
    args = parser.parse_args()

    if args.date:
        validate_date(args.date, data_dir=args.data_dir)
    else:
        validate_all(data_dir=args.data_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
