#!/usr/bin/env python3
"""
ASTRYX INVESTING - Unified CLI
Single entry point for all stock analytics operations

Usage:
    python3 main.py scan                    # Full S&P 500 scan → Excel
    python3 main.py scan --watchlist        # Scan your watchlist only
    python3 main.py scan --top 100          # Scan top 100 only
    python3 main.py scan --date 2026-01-27  # Historical scan

    python3 main.py chart NVDA META         # Generate charts
    python3 main.py chart --watchlist       # Charts for watchlist

    python3 main.py lookup AAPL NVDA        # Fundamental data lookup
    python3 main.py quick AAPL TSLA         # Quick console scan

    python3 main.py fetch                   # Fetch today's fundamentals CSV
    python3 main.py backfill                # Backfill historical CSVs
    python3 main.py validate                # Validate all daily CSVs
    python3 main.py quant                   # Run Value + Growth models
    python3 main.py pipeline                # Full daily pipeline
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    WATCHLIST,
    OUTPUT,
    get_figures_dir,
)


def cmd_scan(args):
    """Run daily market evaluation"""
    from daily_eval import run_daily_evaluation, create_excel_report, print_summary, EXCEL_AVAILABLE
    from data_service import get_sp500_tickers

    # Determine tickers
    if args.tickers:
        tickers = args.tickers
    elif args.watchlist:
        tickers = WATCHLIST
    else:
        tickers = get_sp500_tickers(args.top)

    # Run evaluation
    results = run_daily_evaluation(
        tickers,
        period=args.period,
        eval_date=args.date,
        fetch_fundamentals=not args.no_fundamentals,
    )

    # Print summary
    print_summary(results)

    # Save results
    if not args.no_save:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)

        if args.date:
            timestamp = args.date.replace('-', '')
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d')

        if EXCEL_AVAILABLE:
            excel_path = os.path.join(output_dir, f'daily_eval_{timestamp}.xlsx')
            create_excel_report(results, excel_path)
            print(f"\nSaved: {excel_path}")
        else:
            print("\nInstall openpyxl for Excel export: pip install openpyxl")


def cmd_chart(args):
    """Generate technical analysis charts"""
    from visualizer import plot_stock_analysis
    from datetime import datetime

    # Determine tickers
    if args.watchlist:
        tickers = WATCHLIST
    elif args.tickers:
        tickers = args.tickers
    else:
        print("Error: Specify tickers or use --watchlist")
        return

    base_output = os.path.join(os.path.dirname(__file__), '..', 'output')
    output_dir = get_figures_dir(base_output)

    print(f"\nGenerating charts for: {', '.join(tickers)}")
    print("-" * 50)

    for ticker in tickers:
        try:
            if args.no_save:
                plot_stock_analysis(ticker, args.period)
            else:
                save_path = os.path.join(output_dir, f'{ticker}.png')
                plot_stock_analysis(ticker, args.period, save_path)
            print(f"  {ticker}")
        except Exception as e:
            print(f"  {ticker}: {e}")

    if not args.no_save:
        print(f"\nCharts saved to: {output_dir}")


def cmd_lookup(args):
    """Look up fundamental data for specific tickers"""
    from data_service import DataService
    from config import FUNDAMENTAL_LABELS

    service = DataService()

    for ticker in args.tickers:
        fund = service.get_fundamentals(ticker)

        print(f"\n{'=' * 50}")
        print(f"  {fund.get('name', ticker)} ({ticker})")
        print(f"{'=' * 50}")

        # Identity
        sector = fund.get('sector') or 'Unknown'
        industry = fund.get('industry') or 'Unknown'
        mcap = fund.get('marketCap')
        beta = fund.get('beta')
        print(f"  Sector:     {sector}")
        print(f"  Industry:   {industry}")
        if mcap:
            if mcap >= 1e12:
                print(f"  Market Cap: ${mcap / 1e12:.2f}T")
            else:
                print(f"  Market Cap: ${mcap / 1e9:.1f}B")
        if beta:
            print(f"  Beta:       {beta:.2f}")

        # Valuation
        print(f"\n  VALUATION")
        for field in ['trailingPE', 'forwardPE', 'pegRatio', 'priceToSalesTrailing12Months', 'enterpriseToEbitda']:
            val = fund.get(field)
            label = FUNDAMENTAL_LABELS.get(field, field)
            if val is not None:
                print(f"    {label:16} {val:.2f}")

        # Growth
        print(f"\n  GROWTH")
        for field in ['revenueGrowth', 'earningsGrowth', 'earningsQuarterlyGrowth']:
            val = fund.get(field)
            label = FUNDAMENTAL_LABELS.get(field, field)
            if val is not None:
                print(f"    {label:20} {val * 100:+.1f}%")

        # Profitability
        print(f"\n  PROFITABILITY")
        for field in ['returnOnEquity', 'profitMargins', 'operatingMargins', 'grossMargins']:
            val = fund.get(field)
            label = FUNDAMENTAL_LABELS.get(field, field)
            if val is not None:
                print(f"    {label:18} {val * 100:.1f}%")
        fcf = fund.get('freeCashflow')
        if fcf:
            print(f"    {'Free Cash Flow':18} ${fcf / 1e9:.1f}B")

        # Health
        print(f"\n  HEALTH")
        for field in ['debtToEquity', 'currentRatio']:
            val = fund.get(field)
            label = FUNDAMENTAL_LABELS.get(field, field)
            if val is not None:
                print(f"    {label:18} {val:.2f}")
        for field in ['shortPercentOfFloat', 'shortRatio']:
            val = fund.get(field)
            label = FUNDAMENTAL_LABELS.get(field, field)
            if val is not None:
                if field == 'shortPercentOfFloat':
                    print(f"    {label:18} {val * 100:.1f}%")
                else:
                    print(f"    {label:18} {val:.1f}")

        div = fund.get('dividendYield')
        if div:
            print(f"\n  Dividend Yield:   {div * 100:.2f}%")


def cmd_quick(args):
    """Quick scan of specific tickers (console output only)"""
    from data_service import DataService, fetch_spy_benchmark
    from indicators import add_all_indicators, get_latest_indicators
    from screener import StockScreener

    tickers = args.tickers or WATCHLIST[:6]
    service = DataService()

    print(f"\n=== Quick Scan: {', '.join(tickers)} ===\n")

    # Get SPY for relative strength
    spy_df = fetch_spy_benchmark(args.period)
    if spy_df is not None:
        spy_df = add_all_indicators(spy_df)

    screener = StockScreener()

    for ticker in tickers:
        df = service.fetch_stock(ticker, period=args.period)
        if df is None:
            print(f"{ticker}: No data")
            continue

        try:
            df = add_all_indicators(df, spy_df)
            data = get_latest_indicators(df)
            result = screener.screen_stock(ticker, data)

            print(f"\n{ticker}")
            print(f"  Price: ${data.get('close', 0):.2f}")
            print(f"  RSI:   {data.get('rsi', 0):.1f}")
            print(f"  MACD:  {data.get('macd_hist', 0):.3f}")
            print(f"  Vol:   {data.get('vol_ratio', 0):.2f}x")
            print(f"  RS:    {data.get('rs_vs_spy', 0):+.1f}%")
            print(f"  Signal: {result['signal']}")

            if result['bullish']:
                setups = [s['name'] for s in result['bullish']]
                print(f"  Bullish: {', '.join(setups)}")
            if result['bearish']:
                setups = [s['name'] for s in result['bearish']]
                print(f"  Bearish: {', '.join(setups)}")

        except Exception as e:
            print(f"{ticker}: Error - {e}")


def cmd_fetch(args):
    """Fetch daily S&P 500 fundamentals snapshot"""
    from data_fetcher import run_fetch
    run_fetch(target_date=args.date, top_n=args.top)


def cmd_backfill(args):
    """Backfill historical fundamental CSVs"""
    from historical_backfiller import backfill
    backfill(start_date=args.start, end_date=args.end, skip_existing=not args.force)


def cmd_validate(args):
    """Validate daily CSV snapshots"""
    from validator import validate_all, validate_date
    if args.date:
        validate_date(args.date)
    else:
        validate_all()


def cmd_quant(args):
    """Run quant models (Value Anchor + Growth Aggressor)"""
    from models import run_quant, get_value_score, get_growth_score
    from processor import get_latest_with_trends

    if args.model == 'both':
        run_quant(up_to_date=args.date, save=not args.no_save)
    else:
        df = get_latest_with_trends(up_to_date=args.date)
        if df.empty:
            print("No data available. Run 'python main.py fetch' first!")
            return
        if args.model == 'value':
            picks = get_value_score(df)
        else:
            picks = get_growth_score(df)
        if not picks.empty:
            print(f"\nTop {len(picks)} picks:")
            for _, row in picks.iterrows():
                print(f"  {row['ticker']:6} {row.get('longName', '')[:30]}")


def cmd_pipeline(args):
    """Run full daily pipeline: fetch -> validate -> process -> quant"""
    from datetime import datetime as dt

    target_date = args.date or dt.now().strftime('%Y-%m-%d')
    print(f"\n{'='*50}")
    print(f"  ASTRYX Daily Pipeline: {target_date}")
    print(f"{'='*50}")

    # Step 1: Fetch
    print(f"\n[1/3] Fetching fundamentals...")
    from data_fetcher import run_fetch
    run_fetch(target_date=target_date)

    # Step 2: Validate
    print(f"\n[2/3] Validating data...")
    from validator import validate_date
    result = validate_date(target_date)
    if not result['valid']:
        print("\n  WARNING: Validation failed, proceeding anyway...")

    # Step 3: Run quant models
    print(f"\n[3/3] Running quant models...")
    from models import run_quant
    run_quant(up_to_date=target_date)

    print(f"\n{'='*50}")
    print(f"  Pipeline complete!")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        prog='astryx',
        description='ASTRYX INVESTING - Stock Analytics System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan                    Full S&P 500 scan → Excel report
  python main.py scan --watchlist        Scan watchlist only
  python main.py scan --top 100          Scan top 100 by market cap
  python main.py scan --date 2026-01-27  Historical scan

  python main.py chart NVDA META         Generate charts
  python main.py lookup AAPL NVDA        Fundamental data lookup
  python main.py quick AAPL TSLA         Quick console scan

  python main.py fetch                   Fetch today's S&P 500 fundamentals
  python main.py backfill                Backfill from 2026-01-01 to today
  python main.py validate                Validate all daily CSVs
  python main.py quant                   Run Value + Growth models
  python main.py quant --model value     Run Value Anchor only
  python main.py pipeline                Full daily pipeline
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # -------------------------------------------------------------------------
    # SCAN command
    # -------------------------------------------------------------------------
    scan_parser = subparsers.add_parser('scan', help='Daily market evaluation')
    scan_parser.add_argument('--tickers', '-t', nargs='+', help='Specific tickers')
    scan_parser.add_argument('--watchlist', '-w', action='store_true', help='Use watchlist')
    scan_parser.add_argument('--top', '-n', type=int, default=None, help='Top N S&P 500 (default: all)')
    scan_parser.add_argument('--period', '-p', default='1y', help='Data period (default: 1y)')
    scan_parser.add_argument('--date', '-d', help='Historical date (YYYY-MM-DD)')
    scan_parser.add_argument('--no-save', action='store_true', help="Don't save to file")
    scan_parser.add_argument('--no-fundamentals', action='store_true', help='Skip fundamentals fetch (faster)')
    scan_parser.set_defaults(func=cmd_scan)

    # -------------------------------------------------------------------------
    # CHART command
    # -------------------------------------------------------------------------
    chart_parser = subparsers.add_parser('chart', help='Generate technical charts')
    chart_parser.add_argument('tickers', nargs='*', help='Tickers to chart')
    chart_parser.add_argument('--watchlist', '-w', action='store_true', help='Chart watchlist')
    chart_parser.add_argument('--period', '-p', default='1y', help='Data period (default: 1y)')
    chart_parser.add_argument('--no-save', action='store_true', help='Display instead of save')
    chart_parser.set_defaults(func=cmd_chart)

    # -------------------------------------------------------------------------
    # LOOKUP command
    # -------------------------------------------------------------------------
    lookup_parser = subparsers.add_parser('lookup', help='Fundamental data lookup')
    lookup_parser.add_argument('tickers', nargs='+', help='Tickers to look up')
    lookup_parser.set_defaults(func=cmd_lookup)

    # -------------------------------------------------------------------------
    # QUICK command
    # -------------------------------------------------------------------------
    quick_parser = subparsers.add_parser('quick', help='Quick console scan')
    quick_parser.add_argument('tickers', nargs='*', help='Tickers to scan')
    quick_parser.add_argument('--period', '-p', default='6mo', help='Data period (default: 6mo)')
    quick_parser.set_defaults(func=cmd_quick)

    # -------------------------------------------------------------------------
    # QUANT commands (Dual-Model Engine)
    # -------------------------------------------------------------------------
    # FETCH: Daily fundamental snapshot
    fetch_parser = subparsers.add_parser('fetch', help='Fetch daily S&P 500 fundamentals CSV')
    fetch_parser.add_argument('--date', '-d', default=None, help='Target date (YYYY-MM-DD)')
    fetch_parser.add_argument('--top', '-n', type=int, default=None, help='Only top N tickers')
    fetch_parser.set_defaults(func=cmd_fetch)

    # BACKFILL: Historical data backfill
    backfill_parser = subparsers.add_parser('backfill', help='Backfill historical fundamental CSVs')
    backfill_parser.add_argument('--start', '-s', default='2026-01-01', help='Start date')
    backfill_parser.add_argument('--end', '-e', default=None, help='End date')
    backfill_parser.add_argument('--force', action='store_true', help='Re-fetch existing dates')
    backfill_parser.set_defaults(func=cmd_backfill)

    # VALIDATE: Sanity check CSVs
    validate_parser = subparsers.add_parser('validate', help='Validate daily CSV snapshots')
    validate_parser.add_argument('--date', '-d', default=None, help='Validate specific date')
    validate_parser.set_defaults(func=cmd_validate)

    # QUANT: Run dual-model engine
    quant_parser = subparsers.add_parser('quant', help='Run quant models (Value + Growth)')
    quant_parser.add_argument('--model', '-m', choices=['value', 'growth', 'both'],
                              default='both', help='Which model to run')
    quant_parser.add_argument('--date', '-d', default=None, help='Process data up to this date')
    quant_parser.add_argument('--no-save', action='store_true', help="Don't save recommendations")
    quant_parser.set_defaults(func=cmd_quant)

    # PIPELINE: Run full daily pipeline (fetch + validate + process + quant)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full daily pipeline')
    pipeline_parser.add_argument('--date', '-d', default=None, help='Target date')
    pipeline_parser.set_defaults(func=cmd_pipeline)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
