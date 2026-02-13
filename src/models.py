#!/usr/bin/env python3
"""
ASTRYX INVESTING - Dual-Model Quant Engine
Two specialized scoring models that combine for final recommendations.

Model A: The Value Anchor (defensive)
    - Finds undervalued, high-quality, efficient companies
    - Filters: P/E < 20, Debt/Equity < 1.0
    - Ranks by: ROE, Current Ratio
    - Outputs Top 5 "Undervalued Quality" picks

Model B: The Growth Aggressor (offensive)
    - Finds high-velocity stocks with momentum
    - Filters: Revenue Growth > 15%
    - Ranks by: Operating Margin, Beta
    - Incorporates Short % Float as a "Boost" factor
    - Outputs Top 5 "High-Velocity" picks

Ensemble: 70% current fundamentals / 30% historical trend weighting

Usage:
    python3 models.py                         # Run both models on latest data
    python3 models.py --model value           # Run Value Anchor only
    python3 models.py --model growth          # Run Growth Aggressor only
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from processor import get_latest_with_trends

# Weighting: 70% current-day fundamentals, 30% trend
WEIGHT_CURRENT = 0.70
WEIGHT_TREND = 0.30

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def normalize_series(s: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Min-max normalize a series to [0, 1].
    ascending=True: higher values get higher scores.
    ascending=False: lower values get higher scores (e.g., P/E - lower is better).
    """
    s = pd.to_numeric(s, errors='coerce')
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    s_min, s_max = s.min(), s.max()
    if s_max == s_min:
        return pd.Series(0.5, index=s.index)
    normalized = (s - s_min) / (s_max - s_min)
    if not ascending:
        normalized = 1.0 - normalized
    return normalized


def get_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model A: The Value Anchor
    Finds undervalued, high-quality, efficient companies.

    Filters:
        - Forward P/E < 20 (undervalued)
        - Debt/Equity < 100 (low leverage, stored as ratio * 100 in yfinance)

    Scoring (current fundamentals - 70%):
        - ROE rank (higher is better)
        - Current Ratio rank (higher is better, up to a point)
        - Forward P/E rank (lower is better)
        - Profit Margin rank (higher is better)

    Trend boost (30%):
        - delta_forwardPE < 0 = "Value Compression" = good
        - delta_operatingMargins > 0 = "Efficiency Gains" = good

    Returns top 5 ranked stocks with scores.
    """
    if df.empty:
        return pd.DataFrame()

    # Work on a copy
    scored = df.copy()

    # Filters
    pe_col = 'forwardPE'
    de_col = 'debtToEquity'

    if pe_col in scored.columns:
        scored = scored[scored[pe_col].notna() & (scored[pe_col] > 0) & (scored[pe_col] < 20)]
    if de_col in scored.columns:
        scored = scored[scored[de_col].notna() & (scored[de_col] < 100)]

    if len(scored) < 1:
        print("  Value Anchor: No stocks passed filters, relaxing P/E to < 25")
        scored = df.copy()
        if pe_col in scored.columns:
            scored = scored[scored[pe_col].notna() & (scored[pe_col] > 0) & (scored[pe_col] < 25)]
        if de_col in scored.columns:
            scored = scored[scored[de_col].notna() & (scored[de_col] < 150)]

    if len(scored) < 1:
        print("  Value Anchor: Still no qualifying stocks")
        return pd.DataFrame()

    # Current fundamentals scoring (70%)
    current_score = pd.Series(0.0, index=scored.index)

    if 'returnOnEquity' in scored.columns:
        current_score += normalize_series(scored['returnOnEquity'], ascending=True) * 0.30
    if 'currentRatio' in scored.columns:
        # Cap current ratio at 3.0 (too high can mean capital inefficiency)
        cr = scored['currentRatio'].clip(upper=3.0)
        current_score += normalize_series(cr, ascending=True) * 0.20
    if pe_col in scored.columns:
        current_score += normalize_series(scored[pe_col], ascending=False) * 0.30
    if 'profitMargins' in scored.columns:
        current_score += normalize_series(scored['profitMargins'], ascending=True) * 0.20

    # Trend scoring (30%)
    trend_score = pd.Series(0.0, index=scored.index)
    has_trends = False

    if 'delta_forwardPE' in scored.columns and scored['delta_forwardPE'].notna().any():
        # Negative delta_forwardPE = P/E compressing = good
        trend_score += normalize_series(scored['delta_forwardPE'], ascending=False) * 0.50
        has_trends = True
    if 'delta_operatingMargins' in scored.columns and scored['delta_operatingMargins'].notna().any():
        # Positive delta_operatingMargins = margins expanding = good
        trend_score += normalize_series(scored['delta_operatingMargins'], ascending=True) * 0.50
        has_trends = True

    # Combine
    if has_trends:
        scored['value_score'] = (WEIGHT_CURRENT * current_score) + (WEIGHT_TREND * trend_score)
    else:
        scored['value_score'] = current_score

    # Rank and return top 5
    scored = scored.sort_values('value_score', ascending=False)
    top = scored.head(5).copy()

    # Add model label
    top['model'] = 'Value Anchor'
    top['rank'] = range(1, len(top) + 1)

    return top


def get_growth_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model B: The Growth Aggressor
    Finds high-velocity stocks with accelerating momentum.

    Filters:
        - Revenue Growth > 15%

    Scoring (current fundamentals - 70%):
        - Operating Margin rank (higher is better - efficient growth)
        - Beta rank (higher = more volatile = more upside potential)
        - Revenue Growth rank (higher is better)
        - Short % Float as "Boost" (higher short interest = squeeze potential)

    Trend boost (30%):
        - delta_revenueGrowth > 0 = "Growth Acceleration" = good
        - delta_currentPrice > 0 = "Price Momentum" = good

    Returns top 5 ranked stocks with scores.
    """
    if df.empty:
        return pd.DataFrame()

    scored = df.copy()

    # Filter: Revenue Growth > 15%
    rg_col = 'revenueGrowth'
    if rg_col in scored.columns:
        scored = scored[scored[rg_col].notna() & (scored[rg_col] > 0.15)]

    if len(scored) < 1:
        print("  Growth Aggressor: No stocks > 15% rev growth, relaxing to > 10%")
        scored = df.copy()
        if rg_col in scored.columns:
            scored = scored[scored[rg_col].notna() & (scored[rg_col] > 0.10)]

    if len(scored) < 1:
        print("  Growth Aggressor: Still no qualifying stocks")
        return pd.DataFrame()

    # Current fundamentals scoring (70%)
    current_score = pd.Series(0.0, index=scored.index)

    if 'operatingMargins' in scored.columns:
        current_score += normalize_series(scored['operatingMargins'], ascending=True) * 0.30
    if 'beta' in scored.columns:
        # Cap beta to avoid extreme outliers
        beta = scored['beta'].clip(upper=3.0)
        current_score += normalize_series(beta, ascending=True) * 0.20
    if rg_col in scored.columns:
        current_score += normalize_series(scored[rg_col], ascending=True) * 0.30
    if 'shortPercentOfFloat' in scored.columns:
        # Short squeeze boost - higher short interest = more potential
        short_pct = scored['shortPercentOfFloat'].fillna(0)
        current_score += normalize_series(short_pct, ascending=True) * 0.20

    # Trend scoring (30%)
    trend_score = pd.Series(0.0, index=scored.index)
    has_trends = False

    if 'delta_revenueGrowth' in scored.columns and scored['delta_revenueGrowth'].notna().any():
        trend_score += normalize_series(scored['delta_revenueGrowth'], ascending=True) * 0.50
        has_trends = True
    if 'delta_currentPrice' in scored.columns and scored['delta_currentPrice'].notna().any():
        trend_score += normalize_series(scored['delta_currentPrice'], ascending=True) * 0.50
        has_trends = True

    # Combine
    if has_trends:
        scored['growth_score'] = (WEIGHT_CURRENT * current_score) + (WEIGHT_TREND * trend_score)
    else:
        scored['growth_score'] = current_score

    # Rank and return top 5
    scored = scored.sort_values('growth_score', ascending=False)
    top = scored.head(5).copy()

    top['model'] = 'Growth Aggressor'
    top['rank'] = range(1, len(top) + 1)

    return top


def run_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run both models and produce a consolidated recommendations DataFrame.
    Deduplicates if a stock appears in both (keeps higher-ranked version).
    """
    print("\n--- Model A: Value Anchor ---")
    value_picks = get_value_score(df)
    if not value_picks.empty:
        for _, row in value_picks.iterrows():
            name = row.get('longName', row['ticker'])
            pe = row.get('forwardPE', None)
            roe = row.get('returnOnEquity', None)
            score = row.get('value_score', 0)
            pe_str = f"P/E={pe:.1f}" if pd.notna(pe) else "P/E=N/A"
            roe_str = f"ROE={roe*100:.1f}%" if pd.notna(roe) else "ROE=N/A"
            print(f"  #{int(row['rank'])} {row['ticker']:6} {name[:30]:30} {pe_str:12} {roe_str:12} Score={score:.3f}")
    else:
        print("  No picks")

    print("\n--- Model B: Growth Aggressor ---")
    growth_picks = get_growth_score(df)
    if not growth_picks.empty:
        for _, row in growth_picks.iterrows():
            name = row.get('longName', row['ticker'])
            rev = row.get('revenueGrowth', None)
            beta = row.get('beta', None)
            score = row.get('growth_score', 0)
            rev_str = f"RevG={rev*100:.1f}%" if pd.notna(rev) else "RevG=N/A"
            beta_str = f"Beta={beta:.2f}" if pd.notna(beta) else "Beta=N/A"
            print(f"  #{int(row['rank'])} {row['ticker']:6} {name[:30]:30} {rev_str:14} {beta_str:10} Score={score:.3f}")
    else:
        print("  No picks")

    # Combine
    all_picks = pd.concat([value_picks, growth_picks], ignore_index=True)
    if all_picks.empty:
        return all_picks

    # Deduplicate: if a stock is in both, keep the one with higher combined score
    all_picks['combined_score'] = all_picks.get('value_score', 0).fillna(0) + \
                                   all_picks.get('growth_score', 0).fillna(0)
    all_picks = all_picks.sort_values('combined_score', ascending=False)
    all_picks = all_picks.drop_duplicates(subset='ticker', keep='first')

    return all_picks


def save_recommendations(picks: pd.DataFrame, output_dir: str = None) -> str:
    """
    Save final recommendations to a dated text file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.now().strftime('%Y-%m-%d')
    filepath = os.path.join(output_dir, f'final_recommendations_{today}.txt')

    lines = []
    lines.append(f"ASTRYX INVESTING - Quant Recommendations")
    lines.append(f"Generated: {today}")
    lines.append(f"Strategy: 70% Current Fundamentals / 30% Historical Trend")
    lines.append("=" * 70)

    # Value picks
    value = picks[picks['model'] == 'Value Anchor'] if 'model' in picks.columns else pd.DataFrame()
    lines.append(f"\nMODEL A: VALUE ANCHOR (Top {len(value)} Undervalued Quality)")
    lines.append("-" * 50)
    if not value.empty:
        for _, row in value.iterrows():
            name = row.get('longName', row['ticker'])
            pe = row.get('forwardPE')
            roe = row.get('returnOnEquity')
            de = row.get('debtToEquity')
            cr = row.get('currentRatio')
            score = row.get('value_score', 0)

            lines.append(f"  {row['ticker']:6} {name[:35]:35}")
            metrics = []
            if pd.notna(pe): metrics.append(f"P/E(Fwd)={pe:.1f}")
            if pd.notna(roe): metrics.append(f"ROE={roe*100:.1f}%")
            if pd.notna(de): metrics.append(f"D/E={de:.0f}")
            if pd.notna(cr): metrics.append(f"CR={cr:.2f}")
            lines.append(f"         {', '.join(metrics)}")
            lines.append(f"         Score: {score:.3f}")

            # Trend signals
            delta_pe = row.get('delta_forwardPE')
            delta_margin = row.get('delta_operatingMargins')
            signals = []
            if pd.notna(delta_pe) and delta_pe < -5:
                signals.append(f"Value Compression ({delta_pe:+.1f}%)")
            if pd.notna(delta_margin) and delta_margin > 5:
                signals.append(f"Efficiency Gains ({delta_margin:+.1f}%)")
            if signals:
                lines.append(f"         Trends: {', '.join(signals)}")
            lines.append("")
    else:
        lines.append("  No qualifying stocks")

    # Growth picks
    growth = picks[picks['model'] == 'Growth Aggressor'] if 'model' in picks.columns else pd.DataFrame()
    lines.append(f"\nMODEL B: GROWTH AGGRESSOR (Top {len(growth)} High-Velocity)")
    lines.append("-" * 50)
    if not growth.empty:
        for _, row in growth.iterrows():
            name = row.get('longName', row['ticker'])
            rev = row.get('revenueGrowth')
            beta = row.get('beta')
            margin = row.get('operatingMargins')
            short_pct = row.get('shortPercentOfFloat')
            score = row.get('growth_score', 0)

            lines.append(f"  {row['ticker']:6} {name[:35]:35}")
            metrics = []
            if pd.notna(rev): metrics.append(f"RevG={rev*100:.1f}%")
            if pd.notna(beta): metrics.append(f"Beta={beta:.2f}")
            if pd.notna(margin): metrics.append(f"OpMgn={margin*100:.1f}%")
            if pd.notna(short_pct): metrics.append(f"Short={short_pct*100:.1f}%")
            lines.append(f"         {', '.join(metrics)}")
            lines.append(f"         Score: {score:.3f}")

            # Trend signals
            delta_rev = row.get('delta_revenueGrowth')
            delta_price = row.get('delta_currentPrice')
            signals = []
            if pd.notna(delta_rev) and delta_rev > 5:
                signals.append(f"Growth Acceleration ({delta_rev:+.1f}%)")
            if pd.notna(delta_price) and delta_price > 10:
                signals.append(f"Price Momentum ({delta_price:+.1f}%)")
            if signals:
                lines.append(f"         Trends: {', '.join(signals)}")
            lines.append("")
    else:
        lines.append("  No qualifying stocks")

    lines.append("=" * 70)
    lines.append(f"Total unique picks: {len(picks)}")

    content = '\n'.join(lines)

    with open(filepath, 'w') as f:
        f.write(content)

    return filepath


def run_quant(data_dir: str = None, up_to_date: str = None,
              save: bool = True) -> pd.DataFrame:
    """
    Full quant pipeline: process data -> run models -> save recommendations.
    """
    print("\n=== ASTRYX Quant Engine ===")

    # Get processed data with trends
    df = get_latest_with_trends(data_dir=data_dir, up_to_date=up_to_date)

    if df.empty:
        print("\n  No data available. Run data_fetcher.py first!")
        return pd.DataFrame()

    # Run ensemble
    picks = run_ensemble(df)

    if picks.empty:
        print("\n  No recommendations generated.")
        return picks

    # Save
    if save:
        filepath = save_recommendations(picks)
        print(f"\n  Recommendations saved: {filepath}")

    return picks


def main():
    parser = argparse.ArgumentParser(description='Run quant models')
    parser.add_argument('--model', '-m', choices=['value', 'growth', 'both'],
                        default='both', help='Which model to run')
    parser.add_argument('--date', '-d', default=None,
                        help='Process data up to this date')
    parser.add_argument('--data-dir', default=None,
                        help='Path to raw_snapshots directory')
    parser.add_argument('--no-save', action='store_true',
                        help="Don't save recommendations file")
    args = parser.parse_args()

    df = get_latest_with_trends(data_dir=args.data_dir, up_to_date=args.date)
    if df.empty:
        print("No data available. Run data_fetcher.py first!")
        return

    if args.model == 'value':
        picks = get_value_score(df)
    elif args.model == 'growth':
        picks = get_growth_score(df)
    else:
        picks = run_ensemble(df)

    if not args.no_save and not picks.empty:
        filepath = save_recommendations(picks)
        print(f"\nSaved: {filepath}")


if __name__ == "__main__":
    main()
