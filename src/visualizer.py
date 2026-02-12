#!/usr/bin/env python3
"""
ASTRYX INVESTING - Visualization Module
Creates charts and plots for technical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
from typing import Optional, List, Dict
import yfinance as yf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import add_all_indicators, flatten_columns
from config import get_figures_dir

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_stock_analysis(ticker: str, period: str = "1y", save_path: Optional[str] = None):
    """
    Create a comprehensive stock analysis chart with price, volume, and indicators

    Args:
        ticker: Stock symbol
        period: Data period (e.g., "6mo", "1y", "2y")
        save_path: Optional path to save the figure
    """
    # Fetch data
    df = yf.download(ticker, period=period, progress=False)
    spy = yf.download('SPY', period=period, progress=False)

    if df.empty:
        print(f"No data for {ticker}")
        return

    df = df.reset_index()
    spy = spy.reset_index()
    df = flatten_columns(df)
    spy = flatten_columns(spy)

    # Add indicators
    df = add_all_indicators(df, spy)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.05, figure=fig)

    # Get dates for x-axis
    dates = df['Date']

    # 1. Price chart with moving averages
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, df['Close'], label='Price', linewidth=1.5, color='#2E86AB')
    ax1.plot(dates, df['SMA_20'], label='SMA 20', linewidth=1, alpha=0.8, color='#F6AE2D')
    ax1.plot(dates, df['SMA_50'], label='SMA 50', linewidth=1, alpha=0.8, color='#F26419')
    ax1.plot(dates, df['SMA_200'], label='SMA 200', linewidth=1, alpha=0.8, color='#86BA90')

    # Bollinger Bands
    ax1.fill_between(dates, df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray')

    ax1.set_title(f'{ticker} - Technical Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(dates.iloc[0], dates.iloc[-1])
    ax1.tick_params(labelbottom=False)

    # Add current price annotation
    last_price = df['Close'].iloc[-1]
    ax1.annotate(f'${last_price:.2f}', xy=(dates.iloc[-1], last_price),
                 xytext=(5, 0), textcoords='offset points', fontsize=10, fontweight='bold')

    # 2. Volume
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    colors = ['#86BA90' if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else '#E63946'
              for i in range(1, len(df))]
    colors.insert(0, '#86BA90')
    ax2.bar(dates, df['Volume'], color=colors, alpha=0.7, width=0.8)
    ax2.plot(dates, df['Vol_SMA_20'], color='orange', linewidth=1, label='20-day Avg')
    ax2.set_ylabel('Volume')
    ax2.tick_params(labelbottom=False)
    ax2.legend(loc='upper left', fontsize=8)

    # 3. RSI
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(dates, df['RSI'], color='#7B2CBF', linewidth=1.2)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
    ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.fill_between(dates, 70, 100, alpha=0.1, color='red')
    ax3.fill_between(dates, 0, 30, alpha=0.1, color='green')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.tick_params(labelbottom=False)

    # Add RSI value
    last_rsi = df['RSI'].iloc[-1]
    ax3.annotate(f'{last_rsi:.1f}', xy=(dates.iloc[-1], last_rsi),
                 xytext=(5, 0), textcoords='offset points', fontsize=9)

    # 4. MACD
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(dates, df['MACD'], label='MACD', color='#2E86AB', linewidth=1)
    ax4.plot(dates, df['MACD_Signal'], label='Signal', color='#F6AE2D', linewidth=1)

    # Histogram
    colors_hist = ['#86BA90' if val >= 0 else '#E63946' for val in df['MACD_Hist']]
    ax4.bar(dates, df['MACD_Hist'], color=colors_hist, alpha=0.5, width=0.8)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_ylabel('MACD')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.tick_params(labelbottom=False)

    # 5. Relative Strength vs SPY
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    if 'RS_vs_SPY' in df.columns:
        rs = df['RS_vs_SPY'].fillna(0)
        colors_rs = ['#86BA90' if val >= 0 else '#E63946' for val in rs]
        ax5.fill_between(dates, 0, rs, where=(rs >= 0), color='#86BA90', alpha=0.5)
        ax5.fill_between(dates, 0, rs, where=(rs < 0), color='#E63946', alpha=0.5)
        ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax5.set_ylabel('RS vs SPY (%)')

        # Add RS value
        last_rs = rs.iloc[-1]
        ax5.annotate(f'{last_rs:+.1f}%', xy=(dates.iloc[-1], last_rs),
                     xytext=(5, 0), textcoords='offset points', fontsize=9)

    ax5.set_xlabel('Date')
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # Use constrained_layout instead of tight_layout for sharex axes
    fig.set_constrained_layout(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_screening_results(screen_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a visual summary of screening results

    Args:
        screen_df: DataFrame from screener.screen_multiple()
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Signal Distribution (pie chart)
    ax1 = axes[0, 0]
    signal_counts = screen_df['Signal'].value_counts()
    colors_pie = {'BULLISH': '#86BA90', 'BEARISH': '#E63946', 'NEUTRAL': '#A8DADC',
                  'BULLISH (Mixed)': '#B5E48C', 'BEARISH (Mixed)': '#F4A261', 'CONFLICTING': '#FFD166'}
    pie_colors = [colors_pie.get(s, '#888888') for s in signal_counts.index]
    ax1.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.0f%%',
            colors=pie_colors, startangle=90)
    ax1.set_title('Signal Distribution', fontweight='bold')

    # 2. RSI Distribution
    ax2 = axes[0, 1]
    bullish = screen_df[screen_df['Signal'].str.contains('BULLISH', na=False)]
    bearish = screen_df[screen_df['Signal'].str.contains('BEARISH', na=False)]
    neutral = screen_df[screen_df['Signal'] == 'NEUTRAL']

    if not bullish.empty:
        ax2.hist(bullish['RSI'].dropna(), bins=20, alpha=0.6, label='Bullish', color='#86BA90')
    if not bearish.empty:
        ax2.hist(bearish['RSI'].dropna(), bins=20, alpha=0.6, label='Bearish', color='#E63946')
    if not neutral.empty:
        ax2.hist(neutral['RSI'].dropna(), bins=20, alpha=0.6, label='Neutral', color='#A8DADC')

    ax2.axvline(x=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax2.axvline(x=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax2.set_xlabel('RSI')
    ax2.set_ylabel('Count')
    ax2.set_title('RSI Distribution by Signal', fontweight='bold')
    ax2.legend()

    # 3. Top Bullish Setups (horizontal bar)
    ax3 = axes[1, 0]
    if not bullish.empty:
        top_bullish = bullish.head(10)[['Ticker', 'RSI']].set_index('Ticker')
        colors_bar = ['#86BA90' if rsi < 70 else '#F6AE2D' for rsi in top_bullish['RSI']]
        top_bullish['RSI'].plot(kind='barh', ax=ax3, color=colors_bar)
        ax3.set_xlabel('RSI')
        ax3.set_title('Top 10 Bullish Setups', fontweight='bold')
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, 'No Bullish Setups Found', ha='center', va='center', fontsize=12)
        ax3.set_title('Top 10 Bullish Setups', fontweight='bold')

    # 4. RS vs SPY Scatter
    ax4 = axes[1, 1]
    if 'RS_vs_SPY' in screen_df.columns:
        for signal_type, color in [('BULLISH', '#86BA90'), ('BEARISH', '#E63946'), ('NEUTRAL', '#A8DADC')]:
            mask = screen_df['Signal'].str.contains(signal_type, na=False) if signal_type != 'NEUTRAL' else screen_df['Signal'] == 'NEUTRAL'
            subset = screen_df[mask]
            if not subset.empty:
                ax4.scatter(subset['RS_vs_SPY'], subset['RSI'], alpha=0.6, label=signal_type, color=color, s=50)

        ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Relative Strength vs SPY (%)')
        ax4.set_ylabel('RSI')
        ax4.set_title('RSI vs Relative Strength', fontweight='bold')
        ax4.legend()

    plt.suptitle('ASTRYX INVESTING - Screening Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def create_dashboard_report(tickers: List[str], period: str = "1y",
                            output_dir: str = "output"):
    """
    Create a complete visual report for multiple stocks

    Args:
        tickers: List of stock symbols
        period: Data period
        output_dir: Base output directory (date-based subfolder created automatically)
    """
    figures_dir = get_figures_dir(output_dir)

    print(f"\nGenerating charts for {len(tickers)} stocks...")

    for ticker in tickers:
        try:
            save_path = os.path.join(figures_dir, f'{ticker}.png')
            plot_stock_analysis(ticker, period, save_path)
            print(f"  Created: {ticker}")
        except Exception as e:
            print(f"  Error with {ticker}: {e}")

    print(f"\nCharts saved to: {figures_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ASTRYX INVESTING - Visualization Tools')
    parser.add_argument('--ticker', '-t', default='AAPL', help='Stock ticker to analyze')
    parser.add_argument('--period', '-p', default='1y', help='Data period')
    parser.add_argument('--save', '-s', action='store_true', help='Save chart to file')
    parser.add_argument('--multi', '-m', nargs='+', help='Generate charts for multiple tickers')

    args = parser.parse_args()

    if args.multi:
        create_dashboard_report(args.multi, args.period)
    else:
        if args.save:
            figures_dir = get_figures_dir("output")
            save_path = os.path.join(figures_dir, f'{args.ticker}.png')
        else:
            save_path = None
        plot_stock_analysis(args.ticker, args.period, save_path)
