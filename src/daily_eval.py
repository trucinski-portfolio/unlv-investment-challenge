#!/usr/bin/env python3
"""
Daily Market Evaluation Script for UNLV Investment Challenge

NOTE: Prefer using main.py instead:
    python main.py scan                    # Daily scan → Excel
    python main.py scan --watchlist        # Scan watchlist only
    python main.py scan --date 2026-01-27  # Historical scan
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import argparse
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from centralized modules
from config import (
    WATCHLIST,
    SECTOR_ETFS,
    DEFAULT_PORTFOLIO_VALUE,
    COMPETITION,
    POSITION_SIZING,
    SCREENING,
)
from data_service import (
    DataService,
    get_sp500_tickers,
    fetch_bulk_data,
    fetch_spy_benchmark,
)
from indicators import add_all_indicators, get_latest_indicators
from screener import StockScreener
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


def run_daily_evaluation(tickers: list, period: str = "1y", eval_date: Optional[str] = None) -> dict:
    """
    Run complete daily market evaluation.

    Args:
        tickers: List of ticker symbols to scan
        period: Historical data period (e.g., "1y", "6mo")
        eval_date: Optional date string (YYYY-MM-DD) to get closing prices for.
                   If None, uses most recent available data.

    Returns:
        dict with screening results, setups, sector performance, etc.
    """
    # Initialize data service
    data_service = DataService()

    # Determine the evaluation date
    target_date = None
    if eval_date:
        try:
            target_date = pd.to_datetime(eval_date).date()
            date_display = target_date.strftime('%Y-%m-%d')
            print(f"\n*** HISTORICAL MODE: Getting closing prices for {date_display} ***")
        except ValueError:
            print(f"Invalid date format: {eval_date}. Use YYYY-MM-DD. Falling back to today.")
            eval_date = None
            date_display = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        date_display = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\n" + "=" * 70)
    print("ASTRYX INVESTING - DAILY MARKET EVALUATION")
    print(f"Date: {date_display}")
    print(f"Stocks: {len(tickers)}")
    print("=" * 70)

    results = {
        'run_date': pd.to_datetime(eval_date) if eval_date else datetime.now(),
        'tickers_scanned': len(tickers),
        'eval_date': eval_date
    }

    # 1. Fetch SPY benchmark
    print("\n[1/5] Fetching benchmark data...")
    spy_df = data_service.fetch_spy(period)
    if spy_df is not None:
        if target_date:
            spy_df = data_service.filter_to_date(spy_df, target_date)
            actual_date = pd.to_datetime(spy_df.index[-1]).date()
            if actual_date != target_date:
                print(f"    Note: Using closest available date: {actual_date}")

        spy_df = add_all_indicators(spy_df)
        spy_latest = spy_df.iloc[-1]
        results['spy_price'] = float(spy_latest['Close'])
        results['spy_rsi'] = float(spy_df['RSI'].iloc[-1]) if 'RSI' in spy_df.columns else None

    # 2. Fetch and process all stock data
    print("\n[2/5] Fetching stock data...")
    stock_data = data_service.fetch_multiple(tickers, period=period)

    print("\n[3/5] Calculating indicators...")
    latest_data = {}
    full_data = {}

    for ticker, df in stock_data.items():
        try:
            if target_date:
                df = data_service.filter_to_date(df, target_date)

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
    sector_period = '1mo' if target_date else '5d'

    for etf, name in SECTOR_ETFS.items():
        try:
            etf_df = data_service.fetch_stock(etf, period=sector_period, use_cache=False)
            if etf_df is not None and not etf_df.empty:
                if target_date:
                    etf_df = data_service.filter_to_date(etf_df, target_date)

                current = data_service.get_closing_price(etf_df)
                # Get previous day's close
                if len(etf_df) > 1:
                    prev = data_service.get_closing_price(etf_df.iloc[:-1])
                else:
                    prev = current

                change_pct = ((current - prev) / prev) * 100 if prev > 0 else 0
                sector_data.append({
                    'ETF': etf,
                    'Sector': name,
                    'Price': current,
                    'Change %': change_pct
                })
        except Exception:
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

    # ========== Sheet 6: Decision Helper (NEW!) ==========
    ws_decision = wb.create_sheet("Decision Helper")
    portfolio_value = results.get('portfolio_value', 166600)
    margin_buying_power = portfolio_value * 2  # 2:1 margin

    # Portfolio info header
    gold_fill = PatternFill(start_color='F6AE2D', end_color='F6AE2D', fill_type='solid')
    strong_fill = PatternFill(start_color='2E7D32', end_color='2E7D32', fill_type='solid')
    moderate_fill = PatternFill(start_color='F9A825', end_color='F9A825', fill_type='solid')
    short_fill = PatternFill(start_color='9C27B0', end_color='9C27B0', fill_type='solid')  # Purple for shorts

    ws_decision.cell(row=1, column=1, value=f"DECISION HELPER - Portfolio: ${portfolio_value:,.0f} | Margin Power: ${margin_buying_power:,.0f}")
    ws_decision.cell(row=1, column=1).font = Font(bold=True, size=14)
    ws_decision.merge_cells('A1:L1')

    # Position sizing rules with margin info
    ws_decision.cell(row=2, column=1, value="LONG Rules:")
    ws_decision.cell(row=2, column=2, value=f"Max (25%): ${portfolio_value * 0.25:,.0f}")
    ws_decision.cell(row=2, column=4, value=f"Std (10%): ${portfolio_value * 0.10:,.0f}")
    ws_decision.cell(row=2, column=6, value="Min Buy: $5")
    ws_decision.cell(row=2, column=1).font = Font(bold=True)

    ws_decision.cell(row=3, column=1, value="SHORT Rules:")
    ws_decision.cell(row=3, column=2, value=f"Max (25%): ${portfolio_value * 0.25:,.0f}")
    ws_decision.cell(row=3, column=4, value=f"Std (10%): ${portfolio_value * 0.10:,.0f}")
    ws_decision.cell(row=3, column=6, value="Min Short: $10")
    ws_decision.cell(row=3, column=1).font = Font(bold=True)
    ws_decision.cell(row=3, column=1).fill = short_fill
    ws_decision.cell(row=3, column=1).font = Font(bold=True, color='FFFFFF')

    # Build decision helper data from screening results
    screen_df = results['screening_df'].copy()
    bullish_only = screen_df[screen_df['Signal'].str.contains('BULLISH', na=False)].copy()
    bearish_only = screen_df[screen_df['Signal'].str.contains('BEARISH', na=False)].copy()

    # ===== LONG CANDIDATES =====
    ws_decision.cell(row=5, column=1, value="📈 LONG CANDIDATES (BUY)")
    ws_decision.cell(row=5, column=1).font = Font(bold=True, size=12, color='2E7D32')
    ws_decision.merge_cells('A5:K5')

    if not bullish_only.empty:
        # Calculate conviction score for each stock
        decision_data = []
        for _, row in bullish_only.iterrows():
            score = 0
            reasons = []

            # Scoring criteria
            num_bullish = row.get('Num_Bullish', 0) or 0
            num_bearish = row.get('Num_Bearish', 0) or 0
            rsi = row.get('RSI', 50) or 50
            vol_ratio = row.get('Vol_Ratio', 1) or 1
            rs_spy = row.get('RS_vs_SPY', 0) or 0
            dist_200 = row.get('Dist_200_SMA', 0) or 0
            price = row.get('Price', 0) or 0
            atr_pct = row.get('ATR_Pct', 2) or 2

            # +2 for multiple bullish setups
            if num_bullish >= 2:
                score += 2
                reasons.append("Multi-setup")
            elif num_bullish == 1:
                score += 1

            # -1 for any bearish signals
            if num_bearish > 0:
                score -= 1
                reasons.append("Has bearish")

            # +1 for RSI sweet spot (40-60)
            if 40 <= rsi <= 60:
                score += 1
                reasons.append("RSI sweet spot")
            elif rsi > 70:
                score -= 1
                reasons.append("Overbought")

            # +1 for volume confirmation
            if vol_ratio > 1.2:
                score += 1
                reasons.append("Vol confirm")

            # +1 for relative strength
            if rs_spy > 3:
                score += 1
                reasons.append("RS leader")
            elif rs_spy > 0:
                score += 0.5

            # +1 for healthy trend (not extended)
            if 0 < dist_200 < 15:
                score += 1
                reasons.append("Healthy trend")
            elif dist_200 > 25:
                score -= 1
                reasons.append("Extended")

            # Calculate position size based on score
            if score >= 4:
                position_pct = 0.10  # 10% for high conviction
                conviction = "STRONG"
            elif score >= 2:
                position_pct = 0.07  # 7% for moderate
                conviction = "MODERATE"
            else:
                position_pct = 0.05  # 5% for low
                conviction = "WEAK"

            position_value = portfolio_value * position_pct
            shares = int(position_value / price) if price > 0 else 0
            stop_loss = price * (1 - atr_pct * 2 / 100) if price > 0 else 0
            risk_per_share = price - stop_loss
            total_risk = risk_per_share * shares

            decision_data.append({
                'Ticker': row['Ticker'],
                'Score': score,
                'Conviction': conviction,
                'Price': price,
                'Shares': shares,
                'Position $': position_value,
                'Stop Loss': stop_loss,
                'Risk $': total_risk,
                'RSI': rsi,
                'RS vs SPY': rs_spy,
                'Reasons': ', '.join(reasons[:3])
            })

        # Sort by score descending
        decision_df = pd.DataFrame(decision_data).sort_values('Score', ascending=False)

        # Write headers
        headers = list(decision_df.columns)
        for col_idx, header in enumerate(headers, 1):
            cell = ws_decision.cell(row=6, column=col_idx, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border

        # Write data with color coding
        for row_idx, row in enumerate(decision_df.itertuples(index=False), 7):
            for col_idx, value in enumerate(row, 1):
                col_name = headers[col_idx - 1]

                # Format values
                if col_name in ['Price', 'Stop Loss', 'Position $', 'Risk $']:
                    display_val = f"${value:,.2f}" if pd.notna(value) else ""
                elif col_name in ['RSI', 'RS vs SPY', 'Score']:
                    display_val = f"{value:.1f}" if pd.notna(value) else ""
                else:
                    display_val = value

                cell = ws_decision.cell(row=row_idx, column=col_idx, value=display_val)
                cell.border = border

                # Color code conviction column
                if col_name == 'Conviction':
                    if value == 'STRONG':
                        cell.fill = strong_fill
                        cell.font = Font(bold=True, color='FFFFFF')
                    elif value == 'MODERATE':
                        cell.fill = moderate_fill
                    # WEAK stays default

                # Color code score
                if col_name == 'Score':
                    if float(value) >= 4:
                        cell.fill = bullish_fill
                    elif float(value) < 2:
                        cell.fill = bearish_fill

        long_end_row = 7 + len(decision_df)

        # Set column widths
        col_widths = [10, 8, 12, 10, 8, 12, 10, 10, 8, 10, 25]
        for col_idx, width in enumerate(col_widths, 1):
            ws_decision.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = width

    else:
        ws_decision.cell(row=6, column=1, value="No bullish setups found today - consider waiting for better opportunities")
        long_end_row = 7

    # ===== SHORT CANDIDATES =====
    short_start_row = long_end_row + 2
    ws_decision.cell(row=short_start_row, column=1, value="📉 SHORT CANDIDATES (SELL SHORT)")
    ws_decision.cell(row=short_start_row, column=1).font = Font(bold=True, size=12, color='FFFFFF')
    ws_decision.cell(row=short_start_row, column=1).fill = short_fill
    ws_decision.merge_cells(f'A{short_start_row}:K{short_start_row}')

    if not bearish_only.empty:
        # Calculate conviction score for SHORT positions
        short_data = []
        for _, row in bearish_only.iterrows():
            score = 0
            reasons = []

            # Scoring criteria for SHORTS (inverse of longs)
            num_bullish = row.get('Num_Bullish', 0) or 0
            num_bearish = row.get('Num_Bearish', 0) or 0
            rsi = row.get('RSI', 50) or 50
            vol_ratio = row.get('Vol_Ratio', 1) or 1
            rs_spy = row.get('RS_vs_SPY', 0) or 0
            dist_200 = row.get('Dist_200_SMA', 0) or 0
            price = row.get('Price', 0) or 0
            atr_pct = row.get('ATR_Pct', 2) or 2

            # Check minimum short price ($10)
            if price < 10:
                continue  # Skip stocks under $10 (competition rule)

            # +2 for multiple bearish setups
            if num_bearish >= 2:
                score += 2
                reasons.append("Multi-bearish")
            elif num_bearish == 1:
                score += 1

            # -1 for any bullish signals
            if num_bullish > 0:
                score -= 1
                reasons.append("Has bullish")

            # +1 for RSI overbought (good for shorting)
            if rsi > 70:
                score += 1
                reasons.append("Overbought")
            elif rsi < 30:
                score -= 1
                reasons.append("Oversold-risky")

            # +1 for high volume on down move
            if vol_ratio > 1.2:
                score += 1
                reasons.append("Vol confirm")

            # +1 for relative weakness (underperforming SPY)
            if rs_spy < -3:
                score += 1
                reasons.append("RS weak")
            elif rs_spy < 0:
                score += 0.5

            # +1 for broken trend (below 200 SMA)
            if dist_200 < -5:
                score += 1
                reasons.append("Below 200 SMA")
            elif dist_200 > 15:
                score -= 1  # Too extended up - might squeeze

            # Calculate position size based on score
            if score >= 4:
                position_pct = 0.10
                conviction = "STRONG"
            elif score >= 2:
                position_pct = 0.07
                conviction = "MODERATE"
            else:
                position_pct = 0.05
                conviction = "WEAK"

            position_value = portfolio_value * position_pct
            shares = int(position_value / price) if price > 0 else 0
            # For shorts, stop loss is ABOVE entry (cover if price rises)
            stop_loss = price * (1 + atr_pct * 2 / 100) if price > 0 else 0
            risk_per_share = stop_loss - price
            total_risk = risk_per_share * shares

            short_data.append({
                'Ticker': row['Ticker'],
                'Score': score,
                'Conviction': conviction,
                'Price': price,
                'Shares': shares,
                'Position $': position_value,
                'Stop (Cover)': stop_loss,
                'Risk $': total_risk,
                'RSI': rsi,
                'RS vs SPY': rs_spy,
                'Reasons': ', '.join(reasons[:3])
            })

        if short_data:
            short_df = pd.DataFrame(short_data).sort_values('Score', ascending=False)

            # Write headers
            headers = list(short_df.columns)
            for col_idx, header in enumerate(headers, 1):
                cell = ws_decision.cell(row=short_start_row + 1, column=col_idx, value=header)
                cell.font = header_font
                cell.fill = short_fill
                cell.border = border

            # Write data
            for row_idx, row in enumerate(short_df.itertuples(index=False), short_start_row + 2):
                for col_idx, value in enumerate(row, 1):
                    col_name = headers[col_idx - 1]

                    if col_name in ['Price', 'Stop (Cover)', 'Position $', 'Risk $']:
                        display_val = f"${value:,.2f}" if pd.notna(value) else ""
                    elif col_name in ['RSI', 'RS vs SPY', 'Score']:
                        display_val = f"{value:.1f}" if pd.notna(value) else ""
                    else:
                        display_val = value

                    cell = ws_decision.cell(row=row_idx, column=col_idx, value=display_val)
                    cell.border = border

                    if col_name == 'Conviction':
                        if value == 'STRONG':
                            cell.fill = short_fill
                            cell.font = Font(bold=True, color='FFFFFF')
        else:
            ws_decision.cell(row=short_start_row + 1, column=1, value="No stocks meet short criteria (min $10 price)")
    else:
        ws_decision.cell(row=short_start_row + 1, column=1, value="No bearish setups found today")

    # ========== Sheet 7: Trade Journal ==========
    ws_journal = wb.create_sheet("Trade Journal")

    # Title and instructions
    ws_journal.cell(row=1, column=1, value="TRADE JOURNAL - Track Your Entries & Exits (LONG & SHORT)")
    ws_journal.cell(row=1, column=1).font = Font(bold=True, size=14)
    ws_journal.merge_cells('A1:P1')

    ws_journal.cell(row=2, column=1, value="Instructions: Fill in columns A-I when entering. Fill in J-L when exiting. P&L auto-calculates based on Direction (LONG/SHORT).")
    ws_journal.cell(row=2, column=1).font = Font(italic=True, color='666666')
    ws_journal.merge_cells('A2:P2')

    ws_journal.cell(row=3, column=1, value="Direction: LONG = buy then sell | SHORT = sell then buy to cover. For shorts, profit = entry - exit.")
    ws_journal.cell(row=3, column=1).font = Font(italic=True, color='9C27B0')
    ws_journal.merge_cells('A3:P3')

    # Headers - added Direction column
    journal_headers = [
        'Ticker', 'Direction', 'Setup', 'Entry Date', 'Entry Price', 'Shares', 'Position $',
        'Stop', 'Target', 'Exit Date', 'Exit Price', 'Exit Reason',
        'P&L $', 'P&L %', 'Hold Days', 'Notes'
    ]
    for col_idx, header in enumerate(journal_headers, 1):
        cell = ws_journal.cell(row=5, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border
        cell.alignment = Alignment(horizontal='center')

    # Add Excel formulas for auto-calculation (rows 6-55 for 50 trades)
    # P&L formula accounts for LONG vs SHORT direction
    for row in range(6, 56):
        # P&L $ formula: For LONG: (Exit - Entry) * Shares, For SHORT: (Entry - Exit) * Shares
        pnl_formula = f'=IF(AND(K{row}<>"",E{row}<>"",F{row}<>""),IF(B{row}="SHORT",(E{row}-K{row})*F{row},(K{row}-E{row})*F{row}),"")'
        ws_journal.cell(row=row, column=13, value=pnl_formula)

        # P&L % formula: accounts for direction
        pnl_pct_formula = f'=IF(AND(K{row}<>"",E{row}<>""),IF(B{row}="SHORT",(E{row}-K{row})/E{row}*100,(K{row}-E{row})/E{row}*100),"")'
        ws_journal.cell(row=row, column=14, value=pnl_pct_formula)

        # Hold Days formula: Exit Date - Entry Date
        hold_formula = f'=IF(AND(J{row}<>"",D{row}<>""),J{row}-D{row},"")'
        ws_journal.cell(row=row, column=15, value=hold_formula)

        # Add borders to all cells
        for col in range(1, 17):
            ws_journal.cell(row=row, column=col).border = border

    # Set column widths
    journal_widths = [8, 8, 14, 11, 10, 7, 11, 9, 9, 11, 10, 11, 10, 8, 9, 22]
    for col_idx, width in enumerate(journal_widths, 1):
        ws_journal.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = width

    # Add conditional formatting for P&L column (green if positive, red if negative)
    from openpyxl.formatting.rule import CellIsRule
    green_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
    red_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
    purple_fill = PatternFill(start_color='E1BEE7', end_color='E1BEE7', fill_type='solid')

    # P&L $ column (M) and P&L % column (N) conditional formatting
    ws_journal.conditional_formatting.add('M6:M55',
        CellIsRule(operator='greaterThan', formula=['0'], fill=green_fill))
    ws_journal.conditional_formatting.add('M6:M55',
        CellIsRule(operator='lessThan', formula=['0'], fill=red_fill))
    ws_journal.conditional_formatting.add('N6:N55',
        CellIsRule(operator='greaterThan', formula=['0'], fill=green_fill))
    ws_journal.conditional_formatting.add('N6:N55',
        CellIsRule(operator='lessThan', formula=['0'], fill=red_fill))

    # Direction column - highlight SHORT in purple
    ws_journal.conditional_formatting.add('B6:B55',
        CellIsRule(operator='equal', formula=['"SHORT"'], fill=purple_fill))

    # Summary section at bottom
    ws_journal.cell(row=58, column=1, value="SUMMARY STATS")
    ws_journal.cell(row=58, column=1).font = Font(bold=True, size=12)

    ws_journal.cell(row=59, column=1, value="Total Trades:")
    ws_journal.cell(row=59, column=2, value='=COUNTA(A6:A55)')

    ws_journal.cell(row=60, column=1, value="Long Trades:")
    ws_journal.cell(row=60, column=2, value='=COUNTIF(B6:B55,"LONG")')

    ws_journal.cell(row=61, column=1, value="Short Trades:")
    ws_journal.cell(row=61, column=2, value='=COUNTIF(B6:B55,"SHORT")')
    ws_journal.cell(row=61, column=1).fill = purple_fill

    ws_journal.cell(row=62, column=1, value="Winning Trades:")
    ws_journal.cell(row=62, column=2, value='=COUNTIF(M6:M55,">0")')

    ws_journal.cell(row=63, column=1, value="Losing Trades:")
    ws_journal.cell(row=63, column=2, value='=COUNTIF(M6:M55,"<0")')

    ws_journal.cell(row=64, column=1, value="Win Rate:")
    ws_journal.cell(row=64, column=2, value='=IF(B59>0,B62/B59*100,0)')
    ws_journal.cell(row=64, column=3, value="%")

    ws_journal.cell(row=65, column=1, value="Total P&L:")
    ws_journal.cell(row=65, column=2, value='=SUM(M6:M55)')

    ws_journal.cell(row=66, column=1, value="Avg Hold Days:")
    ws_journal.cell(row=66, column=2, value='=IF(COUNTA(O6:O55)>0,AVERAGE(O6:O55),0)')

    ws_journal.cell(row=67, column=1, value="Avg Win %:")
    ws_journal.cell(row=67, column=2, value='=IF(B62>0,AVERAGEIF(N6:N55,">0"),0)')

    ws_journal.cell(row=68, column=1, value="Avg Loss %:")
    ws_journal.cell(row=68, column=2, value='=IF(B63>0,AVERAGEIF(N6:N55,"<0"),0)')

    # ========== Sheet 8: Historical Log ==========
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
    parser.add_argument(
        '--portfolio', '--pv',
        type=float,
        default=DEFAULT_PORTFOLIO_VALUE,
        help=f'Your portfolio value for position sizing (default: ${DEFAULT_PORTFOLIO_VALUE:,.0f})'
    )
    parser.add_argument(
        '--date', '-d',
        type=str,
        default=None,
        help='Evaluation date (YYYY-MM-DD) to get closing prices for a specific date. '
             'Use this if you missed running the script and want historical data.'
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
    results = run_daily_evaluation(tickers, period=args.period, eval_date=args.date)

    # Add portfolio value to results for Decision Helper sheet
    results['portfolio_value'] = args.portfolio

    # Print summary
    print_summary(results)
    print(f"\nPortfolio Value: ${args.portfolio:,.0f}")
    print(f"  Max Position (25%): ${args.portfolio * 0.25:,.0f}")
    print(f"  Standard (10%): ${args.portfolio * 0.10:,.0f}")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Use the evaluation date for filename if specified, otherwise today
    if args.date:
        timestamp = args.date.replace('-', '')
    else:
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
