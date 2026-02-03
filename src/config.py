"""
ASTRYX INVESTING - Central Configuration
All constants and settings in one place
"""

from typing import Dict, List

# =============================================================================
# COMPETITION RULES (UNLV Spring 2026 Investment Challenge)
# =============================================================================
COMPETITION = {
    'initial_capital': 500_000,
    'max_single_position_pct': 25,    # Max 25% in any single security
    'max_etf_pct': 50,                # Max 50% in ETFs/funds
    'min_buy_price': 5.00,            # Minimum $5 to buy
    'min_short_price': 10.00,         # Minimum $10 to short
    'min_trades': 10,                 # Must make at least 10 trades
    'no_day_trading': True,           # Can't buy and sell same day
}

# Your current portfolio value (update as needed)
DEFAULT_PORTFOLIO_VALUE = 166_600

# =============================================================================
# WATCHLIST & TICKERS
# =============================================================================
WATCHLIST: List[str] = [
    'MSFT', 'META', 'NVDA', 'AAPL', 'TSLA', 'GOOGL',
    'AMZN', 'AMD', 'NFLX', 'CRM', 'AVGO', 'COST'
]

# Top 100 S&P 500 by market cap (for quick scans)
SP500_TOP_100: List[str] = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'ABBV', 'MRK',
    'PEP', 'COST', 'AVGO', 'KO', 'WMT', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT',
    'DHR', 'BAC', 'CRM', 'NKE', 'PFE', 'CMCSA', 'DIS', 'VZ', 'ADBE', 'NFLX',
    'INTC', 'WFC', 'TXN', 'PM', 'RTX', 'NEE', 'T', 'BMY', 'COP', 'ORCL',
    'UPS', 'MS', 'HON', 'QCOM', 'UNP', 'LOW', 'IBM', 'AMD', 'CAT', 'SPGI',
    'GE', 'SBUX', 'BA', 'INTU', 'AMGN', 'DE', 'GS', 'ELV', 'BLK', 'ISRG',
    'GILD', 'AXP', 'LMT', 'MDLZ', 'ADI', 'BKNG', 'TJX', 'MMC', 'SYK', 'CVS',
    'REGN', 'VRTX', 'PLD', 'ADP', 'LRCX', 'CI', 'TMUS', 'MO', 'CB', 'ZTS',
    'SO', 'SCHW', 'DUK', 'CME', 'BDX', 'EOG', 'CL', 'SLB', 'ITW', 'NOC'
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
# POSITION SIZING & RISK MANAGEMENT
# =============================================================================
RISK = {
    'default_risk_pct': 1.0,          # Risk 1% of portfolio per trade
    'stop_loss_atr_mult': 2.0,        # Stop loss = entry - 2*ATR
    'target_atr_mult': 4.0,           # Target = entry + 4*ATR (2:1 R:R)
    'max_positions': 10,              # Max concurrent positions
    'max_hold_days': 20,              # Max days to hold (for backtesting)
}

# Position sizing by conviction level
POSITION_SIZING = {
    'strong': 0.10,      # 10% for high conviction (score >= 4)
    'moderate': 0.07,    # 7% for moderate (score 2-3)
    'weak': 0.05,        # 5% for weak (score < 2)
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
# BACKTEST SETTINGS
# =============================================================================
BACKTEST = {
    'default_period': '2y',
    'position_size_pct': 10,
    'exclude_setups': ['OVERSOLD_BOUNCE'],  # Based on backtest analysis
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
OUTPUT = {
    'excel_date_format': '%Y%m%d',
    'chart_dpi': 150,
    'default_period': '1y',
}
