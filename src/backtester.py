#!/usr/bin/env python3
"""
Backtesting Framework for UNLV Investment Challenge
Tests how screening signals would have performed historically
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass
import yfinance as yf
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import add_all_indicators, get_latest_indicators
from screener import StockScreener, BULLISH_SETUPS, BEARISH_SETUPS


@dataclass
class BacktestTrade:
    """Represents a backtested trade"""
    ticker: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime]
    exit_price: Optional[float]
    setup_type: str
    stop_loss: float
    target: Optional[float]
    shares: int
    status: str  # 'open', 'stopped', 'target', 'time_exit'

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None:
            return 0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    @property
    def r_multiple(self) -> float:
        """Return in terms of R (risk units)"""
        if self.exit_price is None:
            return 0
        risk = self.entry_price - self.stop_loss
        if risk <= 0:
            return 0
        return (self.exit_price - self.entry_price) / risk


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    trades: List[BacktestTrade]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    @property
    def total_return_pct(self) -> float:
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return len([t for t in self.trades if t.pnl > 0])

    @property
    def losing_trades(self) -> int:
        return len([t for t in self.trades if t.pnl < 0])

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def avg_win(self) -> float:
        winners = [t.pnl_pct for t in self.trades if t.pnl > 0]
        return np.mean(winners) if winners else 0

    @property
    def avg_loss(self) -> float:
        losers = [t.pnl_pct for t in self.trades if t.pnl < 0]
        return np.mean(losers) if losers else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss

    @property
    def avg_r_multiple(self) -> float:
        r_multiples = [t.r_multiple for t in self.trades if t.exit_price is not None]
        return np.mean(r_multiples) if r_multiples else 0

    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe calculation based on trade returns"""
        returns = [t.pnl_pct for t in self.trades if t.exit_price is not None]
        if len(returns) < 2:
            return 0
        return (np.mean(returns) / np.std(returns)) * np.sqrt(252 / 20)  # Annualized, assuming ~20 day holds


class Backtester:
    """
    Backtest trading strategies based on screening signals

    Simulates entering positions when setups trigger and
    exiting on stop loss, target, or time-based exit
    """

    def __init__(self,
                 initial_capital: float = 500000,
                 position_size_pct: float = 10,  # % of portfolio per trade
                 stop_loss_atr_mult: float = 2.0,
                 target_atr_mult: float = 4.0,  # 2:1 reward:risk
                 max_hold_days: int = 20,
                 max_positions: int = 10,
                 exclude_setups: List[str] = None):
        """
        Args:
            initial_capital: Starting capital
            position_size_pct: Percentage of portfolio per position
            stop_loss_atr_mult: ATR multiplier for stop loss
            target_atr_mult: ATR multiplier for profit target
            max_hold_days: Maximum days to hold a position
            max_positions: Maximum concurrent positions
            exclude_setups: List of setup names to exclude (e.g., ['OVERSOLD_BOUNCE'])
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.target_atr_mult = target_atr_mult
        self.max_hold_days = max_hold_days
        self.max_positions = max_positions
        self.exclude_setups = exclude_setups or []
        self.screener = StockScreener()

    def _calculate_indicators_for_date(self, df: pd.DataFrame, date_idx: int,
                                        spy_df: pd.DataFrame = None) -> dict:
        """Calculate indicators using only data available up to a specific date"""
        # Get data up to this date
        hist_df = df.iloc[:date_idx + 1].copy()

        if len(hist_df) < 50:  # Need enough data for indicators
            return None

        # Add indicators
        try:
            spy_hist = spy_df.iloc[:date_idx + 1].copy() if spy_df is not None else None
            df_with_ind = add_all_indicators(hist_df, spy_hist)
            return get_latest_indicators(df_with_ind)
        except Exception as e:
            # Silently continue - this happens with insufficient data
            return None

    def backtest_single_stock(self, ticker: str, period: str = "2y",
                               setup_filter: List[str] = None) -> List[BacktestTrade]:
        """
        Backtest a single stock for all setups

        Args:
            ticker: Stock symbol
            period: Historical period to test
            setup_filter: List of setup names to test (None = all bullish)

        Returns:
            List of BacktestTrade objects
        """
        # Fetch data
        try:
            df = yf.download(ticker, period=period, progress=False)
            spy_df = yf.download('SPY', period=period, progress=False)

            if df.empty:
                return []

            df = df.reset_index()
            spy_df = spy_df.reset_index()
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return []

        trades = []
        open_trade = None

        # Calculate ATR for the full dataset (for position sizing reference)
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        # Walk through each day
        for i in range(200, len(df)):  # Start after enough history for 200 SMA

            current_date = df['Date'].iloc[i]

            # Helper to extract scalar from potentially multi-index column
            def get_scalar(series, idx):
                val = series.iloc[idx]
                if hasattr(val, 'iloc'):
                    return float(val.iloc[0])
                return float(val)

            current_price = get_scalar(df['Close'], i)
            current_high = get_scalar(df['High'], i)
            current_low = get_scalar(df['Low'], i)

            atr_val = df['ATR'].iloc[i]
            if hasattr(atr_val, 'iloc'):
                atr_val = atr_val.iloc[0]
            current_atr = float(atr_val) if pd.notna(atr_val) else None

            if current_atr is None:
                continue

            # Check if we have an open trade
            if open_trade is not None:
                days_held = (current_date - open_trade.entry_date).days

                # Check stop loss (using low of day)
                if current_low <= open_trade.stop_loss:
                    open_trade.exit_date = current_date
                    open_trade.exit_price = open_trade.stop_loss
                    open_trade.status = 'stopped'
                    trades.append(open_trade)
                    open_trade = None
                    continue

                # Check target (using high of day)
                if open_trade.target and current_high >= open_trade.target:
                    open_trade.exit_date = current_date
                    open_trade.exit_price = open_trade.target
                    open_trade.status = 'target'
                    trades.append(open_trade)
                    open_trade = None
                    continue

                # Check time exit
                if days_held >= self.max_hold_days:
                    open_trade.exit_date = current_date
                    open_trade.exit_price = current_price
                    open_trade.status = 'time_exit'
                    trades.append(open_trade)
                    open_trade = None
                    continue

            # If no open trade, check for new setup
            if open_trade is None:
                indicators = self._calculate_indicators_for_date(df, i, spy_df)
                if indicators is None:
                    continue

                screen_result = self.screener.screen_stock(ticker, indicators)

                # Check for bullish setups
                if screen_result['bullish']:
                    setup_names = [s['name'] for s in screen_result['bullish']]

                    # Exclude specific setups (e.g., OVERSOLD_BOUNCE which underperformed)
                    if self.exclude_setups:
                        setup_names = [s for s in setup_names if s not in self.exclude_setups]

                    # Apply filter if specified
                    if setup_filter:
                        setup_names = [s for s in setup_names if s in setup_filter]

                    if setup_names:
                        # Calculate position size
                        position_value = self.initial_capital * (self.position_size_pct / 100)
                        shares = int(position_value / current_price)

                        if shares > 0:
                            stop_loss = current_price - (current_atr * self.stop_loss_atr_mult)
                            target = current_price + (current_atr * self.target_atr_mult)

                            open_trade = BacktestTrade(
                                ticker=ticker,
                                entry_date=current_date,
                                entry_price=current_price,
                                exit_date=None,
                                exit_price=None,
                                setup_type=setup_names[0],  # Use first matching setup
                                stop_loss=stop_loss,
                                target=target,
                                shares=shares,
                                status='open'
                            )

        # Close any remaining open trade at last price
        if open_trade is not None:
            if hasattr(df['Close'].iloc[-1], 'iloc'):
                final_price = float(df['Close'].iloc[-1].iloc[0])
            else:
                final_price = float(df['Close'].iloc[-1])
            open_trade.exit_date = df['Date'].iloc[-1]
            open_trade.exit_price = final_price
            open_trade.status = 'time_exit'
            trades.append(open_trade)

        return trades

    def backtest_universe(self, tickers: List[str], period: str = "1y",
                          setup_filter: List[str] = None) -> BacktestResult:
        """
        Backtest across multiple stocks

        Args:
            tickers: List of stock symbols
            period: Historical period
            setup_filter: List of setup names to test

        Returns:
            BacktestResult with aggregated statistics
        """
        all_trades = []

        print(f"\nBacktesting {len(tickers)} stocks over {period}...")

        for ticker in tqdm(tickers, desc="Backtesting"):
            trades = self.backtest_single_stock(ticker, period, setup_filter)
            all_trades.extend(trades)

        # Sort trades by entry date
        all_trades.sort(key=lambda t: t.entry_date)

        # Calculate final capital
        total_pnl = sum(t.pnl for t in all_trades)
        final_capital = self.initial_capital + total_pnl

        # Get date range
        if all_trades:
            start_date = min(t.entry_date for t in all_trades)
            end_date = max(t.exit_date for t in all_trades if t.exit_date)
        else:
            start_date = end_date = datetime.now()

        return BacktestResult(
            trades=all_trades,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital
        )


def print_backtest_report(result: BacktestResult, title: str = "BACKTEST RESULTS"):
    """Print formatted backtest report"""

    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)

    print(f"\nPeriod: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Capital:  ${result.initial_capital:,.2f}")
    print(f"Final Capital:    ${result.final_capital:,.2f}")
    print(f"Total Return:     {result.total_return_pct:+.2f}%")

    print("\n" + "-" * 40)
    print("TRADE STATISTICS:")
    print("-" * 40)
    print(f"Total Trades:     {result.total_trades}")
    print(f"Winning Trades:   {result.winning_trades} ({result.win_rate:.1f}%)")
    print(f"Losing Trades:    {result.losing_trades}")
    print(f"Avg Win:          {result.avg_win:+.2f}%")
    print(f"Avg Loss:         {result.avg_loss:+.2f}%")
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print(f"Avg R-Multiple:   {result.avg_r_multiple:.2f}R")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")

    # Breakdown by setup type
    if result.trades:
        print("\n" + "-" * 40)
        print("BY SETUP TYPE:")
        print("-" * 40)

        setup_stats = {}
        for trade in result.trades:
            if trade.setup_type not in setup_stats:
                setup_stats[trade.setup_type] = {'trades': 0, 'wins': 0, 'pnl': 0}
            setup_stats[trade.setup_type]['trades'] += 1
            if trade.pnl > 0:
                setup_stats[trade.setup_type]['wins'] += 1
            setup_stats[trade.setup_type]['pnl'] += trade.pnl

        for setup, stats in sorted(setup_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            print(f"  {setup:20} | Trades: {stats['trades']:3} | "
                  f"Win Rate: {win_rate:5.1f}% | PnL: ${stats['pnl']:>10,.0f}")

    # Exit type breakdown
    if result.trades:
        print("\n" + "-" * 40)
        print("BY EXIT TYPE:")
        print("-" * 40)

        exit_stats = {}
        for trade in result.trades:
            if trade.status not in exit_stats:
                exit_stats[trade.status] = {'count': 0, 'pnl': 0}
            exit_stats[trade.status]['count'] += 1
            exit_stats[trade.status]['pnl'] += trade.pnl

        for exit_type, stats in exit_stats.items():
            avg_pnl = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {exit_type:12} | Count: {stats['count']:3} | "
                  f"Total PnL: ${stats['pnl']:>10,.0f} | Avg: ${avg_pnl:>8,.0f}")

    print("\n" + "=" * 70)


def run_strategy_comparison(tickers: List[str], period: str = "1y"):
    """Compare different setup strategies"""

    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    strategies = {
        'All Bullish Setups': None,  # None = all setups
        'Oversold Bounce Only': ['OVERSOLD_BOUNCE'],
        'Momentum Breakout Only': ['MOMENTUM_BREAKOUT'],
        'Pullback to Support Only': ['PULLBACK_SUPPORT'],
        'RS Leaders Only': ['RS_LEADER'],
    }

    results = {}
    for name, setup_filter in strategies.items():
        print(f"\nTesting: {name}")
        bt = Backtester(
            initial_capital=500000,
            position_size_pct=10,
            stop_loss_atr_mult=2.0,
            target_atr_mult=4.0,
            max_hold_days=20
        )
        result = bt.backtest_universe(tickers, period, setup_filter)
        results[name] = result

    # Summary comparison
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<30} | {'Return':>8} | {'Win Rate':>8} | {'Trades':>6} | {'Sharpe':>6}")
    print("-" * 70)

    for name, result in results.items():
        print(f"{name:<30} | {result.total_return_pct:>+7.1f}% | "
              f"{result.win_rate:>7.1f}% | {result.total_trades:>6} | "
              f"{result.sharpe_ratio:>6.2f}")

    return results


if __name__ == "__main__":
    # Test the backtester
    from data_collector import get_sp500_tickers
    import argparse

    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--period', '-p', default='2y', help='Backtest period (e.g., 1y, 2y)')
    parser.add_argument('--tickers', '-t', nargs='+',
                        default=['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN',
                                 'TSLA', 'JPM', 'V', 'UNH', 'XOM', 'CVX', 'EOG',
                                 'HD', 'COST', 'RTX', 'GS', 'LIN'],
                        help='Tickers to backtest')
    parser.add_argument('--compare', '-c', action='store_true', help='Run strategy comparison')
    parser.add_argument('--optimized', '-o', action='store_true',
                        help='Use optimized settings (excludes OVERSOLD_BOUNCE)')

    args = parser.parse_args()

    # Based on backtest analysis, excluding OVERSOLD_BOUNCE improves Sharpe from 0.41 to 0.50
    exclude_setups = ['OVERSOLD_BOUNCE'] if args.optimized else []

    print("\n=== UNLV Investment Challenge Backtester ===")
    print(f"Period: {args.period}")
    print(f"Tickers: {len(args.tickers)} stocks")
    print(f"Mode: {'Optimized (excluding OVERSOLD_BOUNCE)' if args.optimized else 'All Setups'}")

    # Single backtest
    bt = Backtester(
        initial_capital=500000,
        position_size_pct=10,
        stop_loss_atr_mult=2.0,
        target_atr_mult=4.0,
        max_hold_days=20,
        exclude_setups=exclude_setups
    )

    result = bt.backtest_universe(args.tickers, period=args.period)
    mode_name = "OPTIMIZED" if args.optimized else "ALL SETUPS"
    print_backtest_report(result, f"BACKTEST: {mode_name} ({args.period.upper()})")

    if args.compare:
        print("\n\nRunning strategy comparison...")
        run_strategy_comparison(args.tickers, period=args.period)
