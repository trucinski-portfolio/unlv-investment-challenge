"""
ASTRYX INVESTING - Central Configuration
All constants and settings in one place
"""

from typing import Dict, List

# =============================================================================
# WATCHLIST & TICKERS
# =============================================================================
WATCHLIST: List[str] = [
    # Tech (core)
    'MSFT', 'META', 'NVDA', 'AAPL', 'TSLA', 'GOOGL',
    'AMZN', 'AMD', 'NFLX', 'CRM', 'AVGO', 'COST',
    # Tech (extended)
    'ADBE', 'NOW', 'SNPS', 'ANET', 'MRVL', 'PANW',
    # Financials
    'JPM', 'V', 'MA', 'GS',
    # Healthcare
    'UNH', 'LLY', 'ISRG',
    # Industrials
    'CAT', 'GE', 'LMT',
    # Energy
    'XOM', 'COP',
    # Consumer
    'HD', 'MCD', 'SBUX',
    # Semis (extra)
    'KLAC', 'LRCX', 'ASML',
]

# Sector ETFs for market overview
SECTOR_ETFS: Dict[str, str] = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLY': 'Consumer Disc.',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication',
}

# =============================================================================
# TECHNICAL INDICATOR SETTINGS
# =============================================================================
INDICATORS = {
    # RSI
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,

    # MACD
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,

    # Moving Averages
    'sma_periods': [10, 20, 50, 200],
    'ema_periods': [9, 21],

    # Bollinger Bands
    'bb_period': 20,
    'bb_std': 2,

    # ATR
    'atr_period': 14,

    # Stochastic
    'stoch_k_period': 14,
    'stoch_d_period': 3,

    # Volume
    'volume_ma_period': 20,
}

# =============================================================================
# SCREENING THRESHOLDS
# =============================================================================
SCREENING = {
    # RSI levels
    'rsi_oversold_entry': 35,
    'rsi_neutral_low': 40,
    'rsi_neutral_high': 60,
    'rsi_overbought_warning': 70,

    # Distance from 200 SMA
    'healthy_trend_min': 0,
    'healthy_trend_max': 15,
    'extended_warning': 25,
    'breakdown_level': -5,

    # Volume confirmation
    'volume_confirm_min': 1.2,
    'volume_surge': 1.5,

    # Relative Strength vs SPY
    'rs_leader_threshold': 3.0,
    'rs_weak_threshold': -3.0,
}

# =============================================================================
# FUNDAMENTAL DATA FIELDS
# =============================================================================
FUNDAMENTALS = {
    'valuation': [
        'trailingPE', 'forwardPE', 'pegRatio',
        'priceToSalesTrailing12Months', 'enterpriseToEbitda',
    ],
    'growth': [
        'revenueGrowth', 'earningsGrowth', 'earningsQuarterlyGrowth',
    ],
    'profitability': [
        'returnOnEquity', 'profitMargins', 'freeCashflow',
        'operatingMargins', 'grossMargins',
    ],
    'health': [
        'debtToEquity', 'currentRatio', 'shortPercentOfFloat', 'shortRatio',
    ],
    'identity': [
        'longName', 'shortName', 'sector', 'industry',
        'marketCap', 'beta', 'averageVolume', 'dividendYield',
    ],
}

# Human-readable labels for fundamental fields
FUNDAMENTAL_LABELS = {
    'trailingPE': 'P/E (TTM)',
    'forwardPE': 'P/E (Fwd)',
    'pegRatio': 'PEG Ratio',
    'priceToSalesTrailing12Months': 'P/S',
    'enterpriseToEbitda': 'EV/EBITDA',
    'revenueGrowth': 'Rev Growth',
    'earningsGrowth': 'Earnings Growth',
    'earningsQuarterlyGrowth': 'Qtr Earnings Growth',
    'returnOnEquity': 'ROE',
    'profitMargins': 'Profit Margin',
    'freeCashflow': 'Free Cash Flow',
    'operatingMargins': 'Operating Margin',
    'grossMargins': 'Gross Margin',
    'debtToEquity': 'Debt/Equity',
    'currentRatio': 'Current Ratio',
    'shortPercentOfFloat': 'Short % Float',
    'shortRatio': 'Short Ratio',
    'marketCap': 'Market Cap',
    'beta': 'Beta',
    'averageVolume': 'Avg Volume',
    'dividendYield': 'Div Yield',
    'sector': 'Sector',
    'industry': 'Industry',
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
OUTPUT = {
    'excel_date_format': '%Y%m%d',
    'chart_dpi': 150,
    'default_period': '1y',
}


def get_figures_dir(base_output_dir: str, date=None) -> str:
    """
    Get the date-based figures directory path (e.g., output/2026.02.11-figures).

    Args:
        base_output_dir: The base output directory (e.g., .../output)
        date: Optional date string (YYYY-MM-DD) or datetime. Defaults to today.

    Returns:
        Path like .../output/2026.02.11-figures
    """
    import os
    from datetime import datetime as dt

    if date is None:
        date_str = dt.now().strftime('%Y.%m.%d')
    elif isinstance(date, str):
        # Convert YYYY-MM-DD to YYYY.MM.DD
        date_str = date.replace('-', '.')
    else:
        date_str = date.strftime('%Y.%m.%d')

    figures_dir = os.path.join(base_output_dir, f'{date_str}-figures')
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir
