#!/usr/bin/env python3
"""
ASTRYX INVESTING - Unified CLI
Single entry point for all trading system operations

Usage:
    python main.py scan                    # Daily market scan → Excel
    python main.py scan --watchlist        # Scan your watchlist only
    python main.py scan --date 2026-01-27  # Historical scan

    python main.py chart NVDA META         # Generate charts
    python main.py chart --watchlist       # Charts for watchlist

    python main.py backtest                # Run strategy backtest
    python main.py backtest --period 2y    # 2-year backtest

    python main.py info AAPL               # Get stock info
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    WATCHLIST,
    DEFAULT_PORTFOLIO_VALUE,
    COMPETITION,
    OUTPUT,
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
        eval_date=args.date
    )

    # Add portfolio value
    results['portfolio_value'] = args.portfolio

    # Print summary
    print_summary(results)
    print(f"\nPortfolio: ${args.portfolio:,.0f}")
    print(f"  Max Position (25%): ${args.portfolio * 0.25:,.0f}")
    print(f"  Standard (10%): ${args.portfolio * 0.10:,.0f}")

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

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'charts')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating charts for: {', '.join(tickers)}")
    print("-" * 50)

    for ticker in tickers:
        try:
            if args.no_save:
                plot_stock_analysis(ticker, args.period)
            else:
                timestamp = datetime.now().strftime('%Y%m%d')
                save_path = os.path.join(output_dir, f'{ticker}_{timestamp}.png')
                plot_stock_analysis(ticker, args.period, save_path)
            print(f"  ✓ {ticker}")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    if not args.no_save:
        print(f"\nCharts saved to: {output_dir}")


def cmd_backtest(args):
    """Run strategy backtest"""
    from backtester import Backtester, print_backtest_report, run_strategy_comparison
    from config import BACKTEST

    tickers = args.tickers or [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN',
        'TSLA', 'JPM', 'V', 'UNH', 'XOM', 'CVX', 'EOG',
        'HD', 'COST', 'RTX', 'GS', 'LIN'
    ]

    exclude_setups = BACKTEST['exclude_setups'] if args.optimized else []

    print("\n=== ASTRYX INVESTING - Backtester ===")
    print(f"Period: {args.period}")
    print(f"Tickers: {len(tickers)} stocks")
    print(f"Mode: {'Optimized' if args.optimized else 'All Setups'}")

    bt = Backtester(
        initial_capital=COMPETITION['initial_capital'],
        position_size_pct=BACKTEST['position_size_pct'],
        exclude_setups=exclude_setups
    )

    result = bt.backtest_universe(tickers, period=args.period)
    mode_name = "OPTIMIZED" if args.optimized else "ALL SETUPS"
    print_backtest_report(result, f"BACKTEST: {mode_name}")

    if args.compare:
        print("\n\nRunning strategy comparison...")
        run_strategy_comparison(tickers, period=args.period)


def cmd_info(args):
    """Get stock information"""
    from data_service import DataService

    service = DataService()

    for ticker in args.tickers:
        info = service.get_stock_info(ticker)
        df = service.fetch_stock(ticker, period='5d')

        print(f"\n{'=' * 40}")
        print(f"{info['name']} ({ticker})")
        print(f"{'=' * 40}")
        print(f"Sector:     {info['sector']}")
        print(f"Industry:   {info['industry']}")
        print(f"Market Cap: ${info['market_cap'] / 1e9:.1f}B" if info['market_cap'] else "N/A")
        print(f"Beta:       {info['beta']:.2f}" if info['beta'] else "N/A")

        if df is not None and not df.empty:
            current_price = service.get_closing_price(df)
            print(f"Price:      ${current_price:.2f}")

        # Position sizing based on portfolio
        portfolio = args.portfolio
        max_pos = portfolio * COMPETITION['max_single_position_pct'] / 100
        std_pos = portfolio * 0.10

        print(f"\nPosition Sizing (${portfolio:,.0f} portfolio):")
        print(f"  Max (25%):      ${max_pos:,.0f}")
        print(f"  Standard (10%): ${std_pos:,.0f}")

        if df is not None:
            current_price = service.get_closing_price(df)
            if current_price > 0:
                print(f"  Max Shares:     {int(max_pos / current_price)}")
                print(f"  Std Shares:     {int(std_pos / current_price)}")


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


def main():
    parser = argparse.ArgumentParser(
        prog='astryx',
        description='ASTRYX INVESTING - Trading Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan                    Daily scan → Excel report
  python main.py scan --watchlist        Scan watchlist only
  python main.py scan --date 2026-01-27  Historical scan

  python main.py chart NVDA META         Generate charts
  python main.py quick AAPL TSLA         Quick console scan

  python main.py backtest                Strategy backtest
  python main.py info AAPL               Stock info + position sizing
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # -------------------------------------------------------------------------
    # SCAN command
    # -------------------------------------------------------------------------
    scan_parser = subparsers.add_parser('scan', help='Daily market evaluation')
    scan_parser.add_argument('--tickers', '-t', nargs='+', help='Specific tickers')
    scan_parser.add_argument('--watchlist', '-w', action='store_true', help='Use watchlist')
    scan_parser.add_argument('--top', '-n', type=int, default=50, help='Top N S&P 500 (default: 50)')
    scan_parser.add_argument('--period', '-p', default='1y', help='Data period (default: 1y)')
    scan_parser.add_argument('--date', '-d', help='Historical date (YYYY-MM-DD)')
    scan_parser.add_argument('--portfolio', type=float, default=DEFAULT_PORTFOLIO_VALUE,
                             help=f'Portfolio value (default: ${DEFAULT_PORTFOLIO_VALUE:,.0f})')
    scan_parser.add_argument('--no-save', action='store_true', help='Don\'t save to file')
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
    # BACKTEST command
    # -------------------------------------------------------------------------
    bt_parser = subparsers.add_parser('backtest', help='Run strategy backtest')
    bt_parser.add_argument('--tickers', '-t', nargs='+', help='Tickers to backtest')
    bt_parser.add_argument('--period', '-p', default='2y', help='Backtest period (default: 2y)')
    bt_parser.add_argument('--optimized', '-o', action='store_true',
                           help='Use optimized settings (excludes underperforming setups)')
    bt_parser.add_argument('--compare', '-c', action='store_true', help='Compare strategies')
    bt_parser.set_defaults(func=cmd_backtest)

    # -------------------------------------------------------------------------
    # INFO command
    # -------------------------------------------------------------------------
    info_parser = subparsers.add_parser('info', help='Get stock info and position sizing')
    info_parser.add_argument('tickers', nargs='+', help='Tickers to look up')
    info_parser.add_argument('--portfolio', type=float, default=DEFAULT_PORTFOLIO_VALUE,
                             help=f'Portfolio value (default: ${DEFAULT_PORTFOLIO_VALUE:,.0f})')
    info_parser.set_defaults(func=cmd_info)

    # -------------------------------------------------------------------------
    # QUICK command
    # -------------------------------------------------------------------------
    quick_parser = subparsers.add_parser('quick', help='Quick console scan')
    quick_parser.add_argument('tickers', nargs='*', help='Tickers to scan')
    quick_parser.add_argument('--period', '-p', default='6mo', help='Data period (default: 6mo)')
    quick_parser.set_defaults(func=cmd_quick)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
