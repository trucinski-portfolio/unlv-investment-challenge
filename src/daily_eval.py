#!/usr/bin/env python3
"""
Daily Market Evaluation Script for UNLV Investment Challenge
Runs a complete market scan and saves results to Excel for tracking

Usage:
    python src/daily_eval.py                    # Full scan with defaults
    python src/daily_eval.py --tickers NVDA AAPL META  # Specific tickers
    python src/daily_eval.py --top 50           # Top 50 S&P 500 stocks
    python src/daily_eval.py --watchlist        # Your watchlist only
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collector import (
    get_sp500_tickers,
    fetch_bulk_data,
    fetch_spy_benchmark,
    SP500_TOP_100,
    SECTOR_ETFS
)
from indicators import add_all_indicators, get_latest_indicators
from screener import StockScreener, generate_screening_report
from position_sizer import PositionSizer

# Try to import openpyxl for Excel support
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.formatting.rule import ColorScaleRule, FormulaRule
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Note: Install openpyxl for Excel support: pip install openpyxl")


# Default watchlist - customize these
WATCHLIST = [
    'MSFT', 'META', 'NVDA', 'AAPL', 'TSLA', 'GOOGL',
    'AMZN', 'AMD', 'NFLX', 'CRM', 'AVGO', 'COST'
]


def run_daily_evaluation(tickers: list, period: str = "1y") -> dict:
    """
    Run complete daily market evaluation

    Returns dict with:
        - summary: Market overview
        - screening: Full screening results
        - setups: Bullish/bearish setups
        - sector_performance: Sector ETF performance
        - position_recommendations: Suggested position sizes
    """
    print("\n" + "=" * 70)
    print("ASTRYX INVESTING - DAILY MARKET EVALUATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stocks: {len(tickers)}")
    print("=" * 70)

    results = {
        'run_date': datetime.now(),
        'tickers_scanned': len(tickers)
    }

    # 1. Fetch SPY benchmark
    print("\n[1/5] Fetching benchmark data...")
    spy_df = fetch_spy_benchmark(period)
    if spy_df is not None:
        spy_df = add_all_indicators(spy_df)
        spy_latest = spy_df.iloc[-1]
        results['spy_price'] = float(spy_latest['Close'])
        results['spy_rsi'] = float(spy_df['RSI'].iloc[-1]) if 'RSI' in spy_df.columns else None

    # 2. Fetch and process all stock data
    print("\n[2/5] Fetching stock data...")
    stock_data = fetch_bulk_data(tickers, period=period)

    print("\n[3/5] Calculating indicators...")
    latest_data = {}
    full_data = {}

    for ticker, df in stock_data.items():
        try:
            df_ind = add_all_indicators(df, spy_df)
            full_data[ticker] = df_ind
            latest_data[ticker] = get_latest_indicators(df_ind)
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")

    print(f"  Processed {len(latest_data)} stocks")

    # 3. Run screener
    print("\n[4/5] Running screener...")
    screener = StockScreener()
    screen_results = screener.screen_multiple(latest_data)
    results['screening_df'] = screen_results

    # 4. Identify setups
    bullish = screen_results[screen_results['Signal'].str.contains('BULLISH', na=False)]
    bearish = screen_results[screen_results['Signal'].str.contains('BEARISH', na=False)]
    neutral = screen_results[screen_results['Signal'] == 'NEUTRAL']

    results['bullish_count'] = len(bullish)
    results['bearish_count'] = len(bearish)
    results['neutral_count'] = len(neutral)
    results['bullish_df'] = bullish
    results['bearish_df'] = bearish

    # 5. Calculate position recommendations for bullish setups
    print("\n[5/5] Calculating position sizes...")
    position_recs = []
    sizer = PositionSizer(portfolio_value=500000)

    for _, row in bullish.head(10).iterrows():
        ticker = row['Ticker']
        if ticker in latest_data:
            data = latest_data[ticker]
            try:
                rec = sizer.calculate_position(
                    ticker=ticker,
                    entry_price=data.get('close', 0),
                    atr=data.get('atr', 0),
                    stop_atr_mult=2.0
                )
                rec['ticker'] = ticker
                rec['signal'] = row['Signal']
                rec['setups'] = row.get('Setups', '')
                rec['rsi'] = data.get('rsi', 0)
                rec['rs_vs_spy'] = data.get('rs_vs_spy', 0)
                position_recs.append(rec)
            except Exception as e:
                pass

    results['positions_df'] = pd.DataFrame(position_recs)

    # 6. Sector performance
    print("\nFetching sector ETF data...")
    sector_data = []
    for etf, name in SECTOR_ETFS.items():
        try:
            import yfinance as yf
            etf_df = yf.download(etf, period='5d', progress=False)
            if not etf_df.empty:
                # Handle multi-index columns
                close_col = etf_df['Close']
                if hasattr(close_col, 'iloc') and hasattr(close_col.iloc[-1], 'iloc'):
                    current = float(close_col.iloc[-1].iloc[0])
                    prev = float(close_col.iloc[-2].iloc[0]) if len(close_col) > 1 else current
                else:
                    current = float(close_col.iloc[-1])
                    prev = float(close_col.iloc[-2]) if len(close_col) > 1 else current

                change_pct = ((current - prev) / prev) * 100
                sector_data.append({
                    'ETF': etf,
                    'Sector': name,
                    'Price': current,
                    'Change %': change_pct
                })
        except Exception as e:
            pass

    results['sectors_df'] = pd.DataFrame(sector_data)

    return results


def create_excel_report(results: dict, output_path: str):
    """
    Create formatted Excel workbook with multiple sheets
    """
    if not EXCEL_AVAILABLE:
        print("Excel export not available. Install openpyxl: pip install openpyxl")
        return None

    wb = openpyxl.Workbook()

    # Define styles
    header_font = Font(bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='2E86AB', end_color='2E86AB', fill_type='solid')
    bullish_fill = PatternFill(start_color='86BA90', end_color='86BA90', fill_type='solid')
    bearish_fill = PatternFill(start_color='E63946', end_color='E63946', fill_type='solid')
    neutral_fill = PatternFill(start_color='A8DADC', end_color='A8DADC', fill_type='solid')
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # ========== Sheet 1: Summary ==========
    ws_summary = wb.active
    ws_summary.title = "Summary"

    summary_data = [
        ["ASTRYX INVESTING - DAILY MARKET EVALUATION"],
        [""],
        ["Run Date:", results['run_date'].strftime('%Y-%m-%d %H:%M:%S')],
        ["Stocks Scanned:", results['tickers_scanned']],
        [""],
        ["MARKET OVERVIEW"],
        ["SPY Price:", f"${results.get('spy_price', 0):.2f}"],
        ["SPY RSI:", f"{results.get('spy_rsi', 0):.1f}" if results.get('spy_rsi') else "N/A"],
        [""],
        ["SIGNAL BREAKDOWN"],
        ["Bullish Setups:", results['bullish_count']],
        ["Bearish Setups:", results['bearish_count']],
        ["Neutral:", results['neutral_count']],
    ]

    for row_idx, row_data in enumerate(summary_data, 1):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws_summary.cell(row=row_idx, column=col_idx, value=value)
            if row_idx == 1:
                cell.font = Font(bold=True, size=14)
            elif value in ["MARKET OVERVIEW", "SIGNAL BREAKDOWN"]:
                cell.font = Font(bold=True)

    ws_summary.column_dimensions['A'].width = 20
    ws_summary.column_dimensions['B'].width = 25

    # ========== Sheet 2: Full Screening Results ==========
    ws_screen = wb.create_sheet("Screening Results")

    screen_df = results['screening_df'].copy()

    # Add headers
    headers = list(screen_df.columns)
    for col_idx, header in enumerate(headers, 1):
        cell = ws_screen.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border
        cell.alignment = Alignment(horizontal='center')

    # Add data
    for row_idx, row in enumerate(screen_df.itertuples(index=False), 2):
        for col_idx, value in enumerate(row, 1):
            cell = ws_screen.cell(row=row_idx, column=col_idx, value=value)
            cell.border = border

            # Color code by signal
            if headers[col_idx - 1] == 'Signal':
                if 'BULLISH' in str(value):
                    cell.fill = bullish_fill
                elif 'BEARISH' in str(value):
                    cell.fill = bearish_fill
                elif value == 'NEUTRAL':
                    cell.fill = neutral_fill

    # Auto-width columns
    for col_idx, header in enumerate(headers, 1):
        ws_screen.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = max(12, len(header) + 2)

    # ========== Sheet 3: Bullish Setups ==========
    ws_bullish = wb.create_sheet("Bullish Setups")

    bullish_df = results['bullish_df'].copy()
    if not bullish_df.empty:
        headers = list(bullish_df.columns)
        for col_idx, header in enumerate(headers, 1):
            cell = ws_bullish.cell(row=1, column=col_idx, value=header)
            cell.font = header_font
            cell.fill = bullish_fill
            cell.border = border

        for row_idx, row in enumerate(bullish_df.itertuples(index=False), 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws_bullish.cell(row=row_idx, column=col_idx, value=value)
                cell.border = border
    else:
        ws_bullish.cell(row=1, column=1, value="No bullish setups found today")

    # ========== Sheet 4: Position Recommendations ==========
    ws_positions = wb.create_sheet("Position Sizes")

    pos_df = results['positions_df']
    if not pos_df.empty:
        # Reorder columns for better readability
        cols_order = ['ticker', 'signal', 'setups', 'entry_price', 'shares',
                      'position_value', 'stop_loss', 'risk_amount', 'rsi', 'rs_vs_spy']
        cols_available = [c for c in cols_order if c in pos_df.columns]
        pos_df = pos_df[cols_available]

        headers = list(pos_df.columns)
        for col_idx, header in enumerate(headers, 1):
            cell = ws_positions.cell(row=1, column=col_idx, value=header.replace('_', ' ').title())
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border

        for row_idx, row in enumerate(pos_df.itertuples(index=False), 2):
            for col_idx, value in enumerate(row, 1):
                # Format numbers appropriately
                if headers[col_idx - 1] in ['entry_price', 'stop_loss', 'position_value', 'risk_amount']:
                    if pd.notna(value):
                        value = f"${value:,.2f}"
                elif headers[col_idx - 1] in ['rsi', 'rs_vs_spy']:
                    if pd.notna(value):
                        value = f"{value:.1f}"

                cell = ws_positions.cell(row=row_idx, column=col_idx, value=value)
                cell.border = border

        for col_idx in range(1, len(headers) + 1):
            ws_positions.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = 15

    # ========== Sheet 5: Sector Performance ==========
    ws_sectors = wb.create_sheet("Sector Performance")

    sector_df = results['sectors_df']
    if not sector_df.empty:
        sector_df = sector_df.sort_values('Change %', ascending=False)

        headers = list(sector_df.columns)
        for col_idx, header in enumerate(headers, 1):
            cell = ws_sectors.cell(row=1, column=col_idx, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border

        for row_idx, row in enumerate(sector_df.itertuples(index=False), 2):
            for col_idx, value in enumerate(row, 1):
                if headers[col_idx - 1] == 'Price':
                    value = f"${value:.2f}"
                elif headers[col_idx - 1] == 'Change %':
                    cell = ws_sectors.cell(row=row_idx, column=col_idx, value=f"{value:+.2f}%")
                    if value > 0:
                        cell.fill = bullish_fill
                    elif value < 0:
                        cell.fill = bearish_fill
                    cell.border = border
                    continue

                cell = ws_sectors.cell(row=row_idx, column=col_idx, value=value)
                cell.border = border

        for col_idx in range(1, len(headers) + 1):
            ws_sectors.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = 20

    # ========== Sheet 6: Historical Log ==========
    ws_log = wb.create_sheet("Daily Log")

    log_headers = ['Date', 'SPY Price', 'SPY RSI', 'Bullish', 'Bearish', 'Neutral', 'Top Setup']
    for col_idx, header in enumerate(log_headers, 1):
        cell = ws_log.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border

    # Add today's entry
    top_setup = results['bullish_df']['Ticker'].iloc[0] if not results['bullish_df'].empty else "None"
    log_data = [
        results['run_date'].strftime('%Y-%m-%d'),
        results.get('spy_price', 0),
        results.get('spy_rsi', 0),
        results['bullish_count'],
        results['bearish_count'],
        results['neutral_count'],
        top_setup
    ]
    for col_idx, value in enumerate(log_data, 1):
        cell = ws_log.cell(row=2, column=col_idx, value=value)
        cell.border = border

    for col_idx in range(1, len(log_headers) + 1):
        ws_log.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = 15

    # Save workbook
    wb.save(output_path)
    return output_path


def save_to_csv_fallback(results: dict, output_dir: str):
    """
    Fallback: Save results to multiple CSV files if Excel not available
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d')

    # Save screening results
    screen_path = os.path.join(output_dir, f'screening_{timestamp}.csv')
    results['screening_df'].to_csv(screen_path, index=False)
    print(f"  Saved: {screen_path}")

    # Save bullish setups
    if not results['bullish_df'].empty:
        bullish_path = os.path.join(output_dir, f'bullish_{timestamp}.csv')
        results['bullish_df'].to_csv(bullish_path, index=False)
        print(f"  Saved: {bullish_path}")

    # Save position recommendations
    if not results['positions_df'].empty:
        pos_path = os.path.join(output_dir, f'positions_{timestamp}.csv')
        results['positions_df'].to_csv(pos_path, index=False)
        print(f"  Saved: {pos_path}")

    # Save sector data
    if not results['sectors_df'].empty:
        sector_path = os.path.join(output_dir, f'sectors_{timestamp}.csv')
        results['sectors_df'].to_csv(sector_path, index=False)
        print(f"  Saved: {sector_path}")

    return output_dir


def print_summary(results: dict):
    """Print summary to console"""
    print("\n" + "=" * 70)
    print("DAILY EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nSPY: ${results.get('spy_price', 0):.2f} | RSI: {results.get('spy_rsi', 0):.1f}")

    print(f"\nSIGNAL BREAKDOWN:")
    print(f"  Bullish: {results['bullish_count']}")
    print(f"  Bearish: {results['bearish_count']}")
    print(f"  Neutral: {results['neutral_count']}")

    if not results['bullish_df'].empty:
        print(f"\nTOP BULLISH SETUPS:")
        for _, row in results['bullish_df'].head(5).iterrows():
            print(f"  {row['Ticker']:6} | RSI: {row['RSI']:.1f} | {row.get('Setups', row['Signal'])}")

    if not results['sectors_df'].empty:
        print(f"\nSECTOR PERFORMANCE (Today):")
        sector_sorted = results['sectors_df'].sort_values('Change %', ascending=False)
        for _, row in sector_sorted.head(5).iterrows():
            print(f"  {row['ETF']:5} ({row['Sector']:20}) | {row['Change %']:+.2f}%")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Daily Market Evaluation - UNLV Investment Challenge'
    )
    parser.add_argument(
        '--tickers', '-t',
        nargs='+',
        help='Specific tickers to scan'
    )
    parser.add_argument(
        '--top', '-n',
        type=int,
        default=50,
        help='Number of top S&P 500 stocks to scan (default: 50)'
    )
    parser.add_argument(
        '--watchlist', '-w',
        action='store_true',
        help='Scan only your watchlist'
    )
    parser.add_argument(
        '--period', '-p',
        default='1y',
        help='Historical data period (default: 1y)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output Excel file path'
    )
    parser.add_argument(
        '--no-excel',
        action='store_true',
        help='Save to CSV instead of Excel'
    )

    args = parser.parse_args()

    # Determine tickers to scan
    if args.tickers:
        tickers = args.tickers
    elif args.watchlist:
        tickers = WATCHLIST
    else:
        tickers = get_sp500_tickers(args.top)

    # Run evaluation
    results = run_daily_evaluation(tickers, period=args.period)

    # Print summary
    print_summary(results)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d')

    if args.no_excel or not EXCEL_AVAILABLE:
        csv_dir = os.path.join(output_dir, 'daily_eval')
        save_to_csv_fallback(results, csv_dir)
        print(f"\nResults saved to CSV files in: {csv_dir}")
    else:
        excel_path = args.output or os.path.join(output_dir, f'daily_eval_{timestamp}.xlsx')
        create_excel_report(results, excel_path)
        print(f"\nExcel report saved to: {excel_path}")

    return results


if __name__ == "__main__":
    main()
