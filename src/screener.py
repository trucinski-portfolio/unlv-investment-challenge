"""
ASTRYX INVESTING - Stock Screener
Identifies trading setups based on technical criteria
"""

import pandas as pd
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class SetupCriteria:
    """Defines criteria for a trading setup"""
    name: str
    description: str
    filter_func: Callable[[dict], bool]
    priority: int = 1  # 1 = highest priority


# ============================================================================
# SETUP DEFINITIONS
# ============================================================================

def is_oversold_bounce(data: dict) -> bool:
    """
    RSI oversold with potential bounce
    - RSI < 35
    - Price above SMA 200 (uptrend context)
    - Volume above average
    """
    rsi = data.get('rsi')
    dist_200 = data.get('dist_sma_200')
    vol_ratio = data.get('vol_ratio')

    if None in (rsi, dist_200, vol_ratio):
        return False

    return (
        rsi < 35 and
        dist_200 > 0 and  # Above 200 SMA
        vol_ratio > 1.0  # Above average volume
    )


def is_momentum_breakout(data: dict) -> bool:
    """
    Strong momentum breakout setup
    - RSI between 50-70 (strength, not overbought)
    - MACD histogram positive and increasing
    - Price above all major MAs
    - Volume spike (>1.5x average)
    """
    rsi = data.get('rsi')
    macd_hist = data.get('macd_hist')
    dist_20 = data.get('dist_sma_20')
    dist_50 = data.get('dist_sma_50')
    dist_200 = data.get('dist_sma_200')
    vol_ratio = data.get('vol_ratio')

    if None in (rsi, macd_hist, dist_20, dist_50, dist_200, vol_ratio):
        return False

    return (
        50 < rsi < 70 and
        macd_hist > 0 and
        dist_20 > 0 and
        dist_50 > 0 and
        dist_200 > 0 and
        vol_ratio > 1.5
    )


def is_pullback_to_support(data: dict) -> bool:
    """
    Pullback to support in uptrend
    - Price pulled back to 20 or 50 SMA
    - Still above 200 SMA (uptrend intact)
    - RSI between 40-55 (not oversold, room to run)
    - MACD still positive
    """
    rsi = data.get('rsi')
    macd = data.get('macd')
    dist_20 = data.get('dist_sma_20')
    dist_50 = data.get('dist_sma_50')
    dist_200 = data.get('dist_sma_200')

    if None in (rsi, macd, dist_20, dist_50, dist_200):
        return False

    near_20sma = -3 < dist_20 < 2  # Within 3% below to 2% above
    near_50sma = -3 < dist_50 < 2

    return (
        (near_20sma or near_50sma) and
        dist_200 > 0 and
        40 < rsi < 55 and
        macd > 0
    )


def is_volume_accumulation(data: dict) -> bool:
    """
    Volume accumulation pattern
    - Increasing volume over recent days
    - Price consolidating (low ATR%)
    - Above 200 SMA
    - RS vs SPY positive
    """
    vol_ratio = data.get('vol_ratio')
    atr_pct = data.get('atr_pct')
    dist_200 = data.get('dist_sma_200')
    rs_spy = data.get('rs_vs_spy')

    if None in (vol_ratio, atr_pct, dist_200):
        return False

    return (
        vol_ratio > 1.2 and
        atr_pct < 3 and  # Low volatility
        dist_200 > 0 and
        (rs_spy is None or rs_spy > 0)  # Outperforming or no data
    )


def is_relative_strength_leader(data: dict) -> bool:
    """
    Relative strength leader vs SPY
    - Outperforming SPY over 20 days
    - RSI > 50 (momentum)
    - Above 50 and 200 SMA
    """
    rs_spy = data.get('rs_vs_spy')
    rsi = data.get('rsi')
    dist_50 = data.get('dist_sma_50')
    dist_200 = data.get('dist_sma_200')

    if None in (rsi, dist_50, dist_200):
        return False

    if rs_spy is None:
        return False

    return (
        rs_spy > 5 and  # Outperforming by >5%
        rsi > 50 and
        dist_50 > 0 and
        dist_200 > 0
    )


def is_overbought_warning(data: dict) -> bool:
    """
    Overbought warning - potential profit taking
    - RSI > 70
    - Extended above 20 SMA (>5%)
    - MACD histogram declining
    """
    rsi = data.get('rsi')
    dist_20 = data.get('dist_sma_20')
    macd_hist = data.get('macd_hist')

    if None in (rsi, dist_20, macd_hist):
        return False

    return (
        rsi > 70 and
        dist_20 > 5
    )


def is_trend_reversal_warning(data: dict) -> bool:
    """
    Potential trend reversal warning
    - Price below 50 SMA
    - MACD negative and declining
    - RSI < 45
    """
    rsi = data.get('rsi')
    macd = data.get('macd')
    macd_hist = data.get('macd_hist')
    dist_50 = data.get('dist_sma_50')

    if None in (rsi, macd, macd_hist, dist_50):
        return False

    return (
        dist_50 < 0 and
        macd < 0 and
        macd_hist < 0 and
        rsi < 45
    )


def is_golden_cross_setup(data: dict) -> bool:
    """
    Near golden cross (50 SMA approaching 200 SMA from below)
    - 50 SMA within 2% of 200 SMA
    - 50 SMA below 200 SMA but price above both
    - RSI > 50
    """
    close = data.get('close')
    sma_50 = data.get('sma_50')
    sma_200 = data.get('sma_200')
    rsi = data.get('rsi')

    if None in (close, sma_50, sma_200, rsi):
        return False

    sma_gap_pct = ((sma_50 - sma_200) / sma_200) * 100

    return (
        -2 < sma_gap_pct < 0 and  # 50 SMA approaching from below
        close > sma_50 and
        close > sma_200 and
        rsi > 50
    )


def is_death_cross_warning(data: dict) -> bool:
    """
    Near death cross warning (50 SMA approaching 200 SMA from above)
    """
    close = data.get('close')
    sma_50 = data.get('sma_50')
    sma_200 = data.get('sma_200')

    if None in (close, sma_50, sma_200):
        return False

    sma_gap_pct = ((sma_50 - sma_200) / sma_200) * 100

    return (
        0 < sma_gap_pct < 2 and  # 50 SMA approaching from above
        close < sma_50
    )


# ============================================================================
# SETUP REGISTRY
# ============================================================================

BULLISH_SETUPS = [
    SetupCriteria(
        name="OVERSOLD_BOUNCE",
        description="RSI oversold (<35) with volume spike in uptrend",
        filter_func=is_oversold_bounce,
        priority=1
    ),
    SetupCriteria(
        name="MOMENTUM_BREAKOUT",
        description="Strong momentum with volume surge above all MAs",
        filter_func=is_momentum_breakout,
        priority=1
    ),
    SetupCriteria(
        name="PULLBACK_SUPPORT",
        description="Healthy pullback to 20/50 SMA in uptrend",
        filter_func=is_pullback_to_support,
        priority=2
    ),
    SetupCriteria(
        name="VOLUME_ACCUMULATION",
        description="Volume building with low volatility consolidation",
        filter_func=is_volume_accumulation,
        priority=2
    ),
    SetupCriteria(
        name="RS_LEADER",
        description="Outperforming SPY with strong technicals",
        filter_func=is_relative_strength_leader,
        priority=2
    ),
    SetupCriteria(
        name="GOLDEN_CROSS",
        description="50 SMA approaching 200 SMA from below",
        filter_func=is_golden_cross_setup,
        priority=3
    ),
]

BEARISH_SETUPS = [
    SetupCriteria(
        name="OVERBOUGHT_WARNING",
        description="RSI >70, extended from 20 SMA",
        filter_func=is_overbought_warning,
        priority=1
    ),
    SetupCriteria(
        name="TREND_REVERSAL",
        description="Below 50 SMA with negative MACD",
        filter_func=is_trend_reversal_warning,
        priority=1
    ),
    SetupCriteria(
        name="DEATH_CROSS",
        description="50 SMA approaching 200 SMA from above",
        filter_func=is_death_cross_warning,
        priority=2
    ),
]


# ============================================================================
# SCREENER CLASS
# ============================================================================

class StockScreener:
    """Screen stocks for trading setups"""

    def __init__(self):
        self.bullish_setups = BULLISH_SETUPS
        self.bearish_setups = BEARISH_SETUPS

    def screen_stock(self, ticker: str, data: dict) -> dict:
        """
        Screen a single stock for all setups

        Args:
            ticker: Stock symbol
            data: Dictionary of indicator values

        Returns:
            Dictionary with found setups
        """
        result = {
            'ticker': ticker,
            'bullish': [],
            'bearish': [],
            'signal': 'NEUTRAL'
        }

        # Check bullish setups
        for setup in self.bullish_setups:
            if setup.filter_func(data):
                result['bullish'].append({
                    'name': setup.name,
                    'description': setup.description,
                    'priority': setup.priority
                })

        # Check bearish setups
        for setup in self.bearish_setups:
            if setup.filter_func(data):
                result['bearish'].append({
                    'name': setup.name,
                    'description': setup.description,
                    'priority': setup.priority
                })

        # Determine overall signal
        if result['bullish'] and not result['bearish']:
            result['signal'] = 'BULLISH'
        elif result['bearish'] and not result['bullish']:
            result['signal'] = 'BEARISH'
        elif result['bullish'] and result['bearish']:
            # Priority 1 setups override
            bull_priority = min([s['priority'] for s in result['bullish']])
            bear_priority = min([s['priority'] for s in result['bearish']])
            if bull_priority < bear_priority:
                result['signal'] = 'BULLISH (Mixed)'
            elif bear_priority < bull_priority:
                result['signal'] = 'BEARISH (Mixed)'
            else:
                result['signal'] = 'CONFLICTING'

        return result

    def screen_multiple(self, stocks_data: dict) -> pd.DataFrame:
        """
        Screen multiple stocks

        Args:
            stocks_data: Dictionary of {ticker: indicator_dict}

        Returns:
            DataFrame with screening results
        """
        results = []

        for ticker, data in stocks_data.items():
            screen_result = self.screen_stock(ticker, data)

            # Flatten for DataFrame
            row = {
                'Ticker': ticker,
                'Signal': screen_result['signal'],
                'Bullish_Setups': ', '.join([s['name'] for s in screen_result['bullish']]),
                'Bearish_Setups': ', '.join([s['name'] for s in screen_result['bearish']]),
                'Num_Bullish': len(screen_result['bullish']),
                'Num_Bearish': len(screen_result['bearish']),
                # Add key metrics for sorting
                'Price': data.get('close'),
                'RSI': data.get('rsi'),
                'MACD_Hist': data.get('macd_hist'),
                'Vol_Ratio': data.get('vol_ratio'),
                'RS_vs_SPY': data.get('rs_vs_spy'),
                'Dist_200_SMA': data.get('dist_sma_200'),
                'ATR_Pct': data.get('atr_pct'),
            }
            results.append(row)

        df = pd.DataFrame(results)

        # Sort by signal strength
        signal_order = {'BULLISH': 0, 'BULLISH (Mixed)': 1, 'NEUTRAL': 2,
                        'CONFLICTING': 3, 'BEARISH (Mixed)': 4, 'BEARISH': 5}
        df['Signal_Order'] = df['Signal'].map(signal_order)
        df = df.sort_values(['Signal_Order', 'Num_Bullish'], ascending=[True, False])
        df = df.drop('Signal_Order', axis=1)

        return df


def generate_screening_report(screen_df: pd.DataFrame) -> str:
    """Generate a text report from screening results"""

    report = []
    report.append("=" * 60)
    report.append("ASTRYX INVESTING - STOCK SCREENING REPORT")
    report.append("=" * 60)
    report.append("")

    # Bullish opportunities
    bullish = screen_df[screen_df['Signal'].str.contains('BULLISH', na=False)]
    if not bullish.empty:
        report.append("📈 BULLISH SETUPS FOUND:")
        report.append("-" * 40)
        for _, row in bullish.iterrows():
            report.append(f"  {row['Ticker']:6} | RSI: {row['RSI']:.1f} | "
                         f"Setups: {row['Bullish_Setups']}")
        report.append("")

    # Bearish warnings
    bearish = screen_df[screen_df['Signal'].str.contains('BEARISH', na=False)]
    if not bearish.empty:
        report.append("📉 BEARISH WARNINGS:")
        report.append("-" * 40)
        for _, row in bearish.iterrows():
            report.append(f"  {row['Ticker']:6} | RSI: {row['RSI']:.1f} | "
                         f"Setups: {row['Bearish_Setups']}")
        report.append("")

    # Summary stats
    report.append("SUMMARY:")
    report.append("-" * 40)
    report.append(f"  Total Screened: {len(screen_df)}")
    report.append(f"  Bullish Setups: {len(bullish)}")
    report.append(f"  Bearish Setups: {len(bearish)}")
    report.append(f"  Neutral: {len(screen_df[screen_df['Signal'] == 'NEUTRAL'])}")
    report.append("")

    # Top RS leaders
    if 'RS_vs_SPY' in screen_df.columns:
        rs_leaders = screen_df.nlargest(5, 'RS_vs_SPY')[['Ticker', 'RS_vs_SPY']]
        if not rs_leaders.empty:
            report.append("TOP RELATIVE STRENGTH vs SPY:")
            report.append("-" * 40)
            for _, row in rs_leaders.iterrows():
                rs_val = row['RS_vs_SPY']
                if pd.notna(rs_val):
                    report.append(f"  {row['Ticker']:6} | RS: {rs_val:+.1f}%")
            report.append("")

    report.append("=" * 60)

    return "\n".join(report)


if __name__ == "__main__":
    # Test with sample data
    print("=== Stock Screener Test ===\n")

    # Sample data
    test_data = {
        'AAPL': {
            'close': 180, 'rsi': 32, 'macd': 0.5, 'macd_hist': 0.1,
            'dist_sma_20': -1, 'dist_sma_50': 2, 'dist_sma_200': 10,
            'vol_ratio': 1.3, 'rs_vs_spy': 3, 'atr_pct': 2.1,
            'sma_50': 175, 'sma_200': 165
        },
        'TSLA': {
            'close': 250, 'rsi': 75, 'macd': 5, 'macd_hist': -0.5,
            'dist_sma_20': 8, 'dist_sma_50': 12, 'dist_sma_200': 25,
            'vol_ratio': 1.8, 'rs_vs_spy': 8, 'atr_pct': 4.5,
            'sma_50': 220, 'sma_200': 200
        }
    }

    screener = StockScreener()
    results = screener.screen_multiple(test_data)

    print(results[['Ticker', 'Signal', 'Bullish_Setups', 'Bearish_Setups']])
    print("\n")
    print(generate_screening_report(results))
