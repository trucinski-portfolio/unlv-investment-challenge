#!/usr/bin/env python3
"""
UNLV Investment Challenge - Trading Dashboard

DEPRECATED: Use main.py instead:
    python main.py scan         # Daily market scan
    python main.py chart NVDA   # Generate charts
    python main.py backtest     # Run backtest

This file is kept for backward compatibility but will be removed.
"""

import pandas as pd
from datetime import datetime
import argparse
import os
import sys
import warnings

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import WATCHLIST
from data_service import (
    get_sp500_tickers,
    fetch_bulk_data,
    fetch_spy_benchmark,
    fetch_stock_metadata,
)
from indicators import add_all_indicators, get_latest_indicators
from screener import StockScreener, generate_screening_report
from visualizer import (
    plot_stock_analysis,
    plot_screening_results,
    plot_backtest_results,
    plot_portfolio_allocation,
    create_dashboard_report
)


def run_full_scan(num_stocks: int = 100, period: str = "2y", save_csv: bool = True):
    """
    Run a full scan of S&P 500 stocks

    Args:
        num_stocks: Number of top S&P 500 stocks to scan
        period: Historical data period
        save_csv: Whether to save results to CSV
    """
    print("\n" + "=" * 70)
    print("ASTRYX INVESTING - UNLV INVESTMENT CHALLENGE")
    print("Stock Screening Dashboard")
    print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Get tickers
    tickers = get_sp500_tickers(num_stocks)
    print(f"\nScanning top {len(tickers)} S&P 500 stocks...")

    # 2. Fetch SPY benchmark first
    print("\nFetching SPY benchmark data...")
    spy_df = fetch_spy_benchmark(period)
    if spy_df is not None:
        spy_df = add_all_indicators(spy_df)
        print(f"  SPY: {len(spy_df)} trading days loaded")

    # 3. Fetch all stock data
    stock_data = fetch_bulk_data(tickers, period=period)

    # 4. Calculate indicators for each stock
    print("\nCalculating technical indicators...")
    stocks_with_indicators = {}
    latest_data = {}

    for ticker, df in stock_data.items():
        try:
            # Add all indicators
            df_ind = add_all_indicators(df, spy_df)
            stocks_with_indicators[ticker] = df_ind

            # Get latest values for screening
            latest = get_latest_indicators(df_ind)
            latest_data[ticker] = latest
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")

    print(f"  Processed {len(latest_data)} stocks successfully")

    # 5. Run screener
    print("\nRunning screener...")
    screener = StockScreener()
    screen_results = screener.screen_multiple(latest_data)

    # 6. Generate report
    report = generate_screening_report(screen_results)
    print("\n" + report)

    # 7. Save results
    if save_csv:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save screening results
        screen_file = os.path.join(output_dir, f'screening_{timestamp}.csv')
        screen_results.to_csv(screen_file, index=False)
        print(f"\nScreening results saved to: {screen_file}")

        # Save detailed data for bullish setups
        bullish_tickers = screen_results[
            screen_results['Signal'].str.contains('BULLISH', na=False)
        ]['Ticker'].tolist()

        if bullish_tickers:
            detailed_file = os.path.join(output_dir, f'bullish_detailed_{timestamp}.csv')
            detailed_rows = []
            for ticker in bullish_tickers:
                if ticker in latest_data:
                    row = latest_data[ticker].copy()
                    row['ticker'] = ticker
                    detailed_rows.append(row)
            pd.DataFrame(detailed_rows).to_csv(detailed_file, index=False)
            print(f"Bullish detailed data saved to: {detailed_file}")

    return screen_results, latest_data


def run_quick_scan(tickers: list, period: str = "6mo"):
    """
    Run a quick scan on specific tickers

    Args:
        tickers: List of ticker symbols
        period: Historical data period
    """
    print("\n" + "=" * 50)
    print("QUICK SCAN")
    print(f"Tickers: {', '.join(tickers)}")
    print("=" * 50)

    # Fetch SPY
    spy_df = fetch_spy_benchmark(period)
    if spy_df is not None:
        spy_df = add_all_indicators(spy_df)

    # Fetch and process each ticker
    stock_data = fetch_bulk_data(tickers, period=period)
    latest_data = {}

    for ticker, df in stock_data.items():
        try:
            df_ind = add_all_indicators(df, spy_df)
            latest_data[ticker] = get_latest_indicators(df_ind)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Screen
    screener = StockScreener()
    results = screener.screen_multiple(latest_data)

    # Print detailed results
    print("\n" + "-" * 50)
    for ticker in tickers:
        if ticker in latest_data:
            data = latest_data[ticker]
            screen = screener.screen_stock(ticker, data)

            print(f"\n{ticker}:")
            print(f"  Price: ${data.get('close', 0):.2f}")
            print(f"  RSI: {data.get('rsi', 0):.1f}")
            print(f"  MACD Histogram: {data.get('macd_hist', 0):.3f}")
            print(f"  Volume Ratio: {data.get('vol_ratio', 0):.2f}x avg")
            print(f"  Dist from 200 SMA: {data.get('dist_sma_200', 0):.1f}%")
            print(f"  RS vs SPY (20d): {data.get('rs_vs_spy', 0):.1f}%")
            print(f"  Signal: {screen['signal']}")
            if screen['bullish']:
                print(f"  Bullish: {', '.join([s['name'] for s in screen['bullish']])}")
            if screen['bearish']:
                print(f"  Bearish: {', '.join([s['name'] for s in screen['bearish']])}")

    return results, latest_data


def get_top_setups(screen_results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get top N bullish setups"""
    bullish = screen_results[screen_results['Signal'].str.contains('BULLISH', na=False)]
    return bullish.head(n)


def run_chart_mode(tickers: list, period: str = "1y", save_charts: bool = True):
    """
    Generate technical analysis charts for specified tickers

    Args:
        tickers: List of ticker symbols
        period: Historical data period
        save_charts: Whether to save charts to files
    """
    print("\n" + "=" * 70)
    print("ASTRYX INVESTING - CHART MODE")
    print(f"Generating charts for: {', '.join(tickers)}")
    print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'charts')
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tickers:
        try:
            if save_charts:
                timestamp = datetime.now().strftime('%Y%m%d')
                save_path = os.path.join(output_dir, f'{ticker}_{timestamp}.png')
                plot_stock_analysis(ticker, period, save_path)
            else:
                plot_stock_analysis(ticker, period)
            print(f"  ✓ {ticker} chart created")
        except Exception as e:
            print(f"  ✗ {ticker} error: {e}")

    if save_charts:
        print(f"\nCharts saved to: {output_dir}")


def run_full_scan_with_charts(num_stocks: int = 100, period: str = "2y",
                              save_csv: bool = True, generate_charts: bool = True):
    """
    Run a full scan and optionally generate charts for bullish setups
    """
    # Run the standard full scan
    screen_results, latest_data = run_full_scan(num_stocks, period, save_csv)

    # Generate charts for bullish setups
    if generate_charts:
        bullish_tickers = screen_results[
            screen_results['Signal'].str.contains('BULLISH', na=False)
        ]['Ticker'].tolist()[:10]  # Top 10 bullish

        if bullish_tickers:
            print("\n" + "-" * 50)
            print("Generating charts for top bullish setups...")
            run_chart_mode(bullish_tickers, period)

            # Also create screening summary chart
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'charts')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_path = os.path.join(output_dir, f'screening_summary_{timestamp}.png')
            plot_screening_results(screen_results, summary_path)

    return screen_results, latest_data


def main():
    # Deprecation warning
    print("\n" + "=" * 60)
    print("WARNING: dashboard.py is deprecated. Use main.py instead:")
    print("  python main.py scan         # Daily market scan")
    print("  python main.py chart NVDA   # Generate charts")
    print("  python main.py quick AAPL   # Quick console scan")
    print("=" * 60 + "\n")

    parser = argparse.ArgumentParser(
        description='DEPRECATED - Use main.py instead'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['full', 'quick', 'watchlist', 'chart'],
        default='quick',
        help='Scan mode: full (100 stocks), quick (test), watchlist (your picks), chart (generate charts)'
    )
    parser.add_argument(
        '--tickers', '-t',
        nargs='+',
        default=['AAPL', 'MSFT', 'NVDA', 'META', 'TSLA', 'GOOGL'],
        help='Tickers for quick scan or chart mode'
    )
    parser.add_argument(
        '--num-stocks', '-n',
        type=int,
        default=50,
        help='Number of S&P 500 stocks to scan in full mode'
    )
    parser.add_argument(
        '--period', '-p',
        default='1y',
        help='Historical data period (e.g., 6mo, 1y, 2y)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to CSV/charts'
    )
    parser.add_argument(
        '--charts',
        action='store_true',
        help='Generate charts for bullish setups (in full mode)'
    )

    args = parser.parse_args()

    if args.mode == 'full':
        if args.charts:
            run_full_scan_with_charts(
                num_stocks=args.num_stocks,
                period=args.period,
                save_csv=not args.no_save,
                generate_charts=True
            )
        else:
            run_full_scan(
                num_stocks=args.num_stocks,
                period=args.period,
                save_csv=not args.no_save
            )
    elif args.mode == 'watchlist':
        run_quick_scan(WATCHLIST, period=args.period)
        if args.charts:
            run_chart_mode(WATCHLIST, period=args.period, save_charts=not args.no_save)
    elif args.mode == 'chart':
        # Chart-only mode
        run_chart_mode(args.tickers, period=args.period, save_charts=not args.no_save)
    else:  # quick
        run_quick_scan(args.tickers, period=args.period)
        if args.charts:
            run_chart_mode(args.tickers, period=args.period, save_charts=not args.no_save)


if __name__ == "__main__":
    main()
