"""
ASTRYX INVESTING - Technical Indicators Module
Calculates RSI, MACD, volume trends, relative strength, and more
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index

    Args:
        prices: Series of closing prices
        period: RSI period (default 14)

    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series,
                   fast: int = 12,
                   slow: int = 26,
                   signal: int = 9) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_moving_averages(prices: pd.Series) -> dict:
    """
    Calculate common moving averages

    Returns:
        Dictionary with SMA and EMA values
    """
    return {
        'SMA_10': prices.rolling(window=10).mean(),
        'SMA_20': prices.rolling(window=20).mean(),
        'SMA_50': prices.rolling(window=50).mean(),
        'SMA_200': prices.rolling(window=200).mean(),
        'EMA_9': prices.ewm(span=9, adjust=False).mean(),
        'EMA_21': prices.ewm(span=21, adjust=False).mean(),
    }


def calculate_volume_indicators(df: pd.DataFrame) -> dict:
    """
    Calculate volume-based indicators

    Args:
        df: DataFrame with 'Close' and 'Volume' columns

    Returns:
        Dictionary with volume indicators
    """
    volume = df['Volume']
    close = df['Close']

    # Volume moving averages
    vol_sma_20 = volume.rolling(window=20).mean()
    vol_sma_50 = volume.rolling(window=50).mean()

    # Volume ratio (current vs 20-day average)
    vol_ratio = volume / vol_sma_20

    # On-Balance Volume (OBV)
    obv = (np.sign(close.diff()) * volume).cumsum()

    # Volume Weighted Average Price (VWAP) - daily approximation
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()

    return {
        'Vol_SMA_20': vol_sma_20,
        'Vol_SMA_50': vol_sma_50,
        'Vol_Ratio': vol_ratio,  # >1 means above average volume
        'OBV': obv,
        'VWAP': vwap
    }


def calculate_relative_strength(stock_prices: pd.Series,
                                benchmark_prices: pd.Series,
                                period: int = 20) -> pd.Series:
    """
    Calculate relative strength vs benchmark (typically SPY)

    Args:
        stock_prices: Series of stock closing prices
        benchmark_prices: Series of benchmark closing prices
        period: Period for rate of change calculation

    Returns:
        Series of relative strength values
    """
    # Align the series
    stock_aligned = stock_prices.reindex(benchmark_prices.index, method='ffill')
    bench_aligned = benchmark_prices.reindex(stock_prices.index, method='ffill')

    # Calculate rate of change for both
    stock_roc = stock_aligned.pct_change(period) * 100
    bench_roc = bench_aligned.pct_change(period) * 100

    # Relative strength = stock performance - benchmark performance
    relative_strength = stock_roc - bench_roc

    return relative_strength


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (for volatility/position sizing)

    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period (default 14)

    Returns:
        Series of ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_bollinger_bands(prices: pd.Series,
                               period: int = 20,
                               std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands

    Returns:
        Tuple of (upper band, middle band, lower band, %B)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    # %B indicator (where price is relative to bands)
    percent_b = (prices - lower) / (upper - lower)

    return upper, middle, lower, percent_b


def calculate_stochastic(df: pd.DataFrame,
                          k_period: int = 14,
                          d_period: int = 3) -> tuple:
    """
    Calculate Stochastic Oscillator

    Returns:
        Tuple of (%K, %D)
    """
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()

    stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns from yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def add_all_indicators(df: pd.DataFrame,
                        benchmark_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add all technical indicators to a stock dataframe

    Args:
        df: DataFrame with OHLCV data
        benchmark_df: Optional benchmark DataFrame for relative strength

    Returns:
        DataFrame with all indicators added
    """
    result = flatten_columns(df.copy())
    if benchmark_df is not None:
        benchmark_df = flatten_columns(benchmark_df.copy())

    close = result['Close']

    # RSI
    result['RSI'] = calculate_rsi(close)

    # MACD
    macd, signal, hist = calculate_macd(close)
    result['MACD'] = macd
    result['MACD_Signal'] = signal
    result['MACD_Hist'] = hist

    # Moving Averages
    mas = calculate_moving_averages(close)
    for name, values in mas.items():
        result[name] = values

    # Volume indicators
    vol_indicators = calculate_volume_indicators(result)
    for name, values in vol_indicators.items():
        result[name] = values

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_pct = calculate_bollinger_bands(close)
    result['BB_Upper'] = bb_upper
    result['BB_Middle'] = bb_middle
    result['BB_Lower'] = bb_lower
    result['BB_Pct'] = bb_pct

    # ATR
    result['ATR'] = calculate_atr(result)
    result['ATR_Pct'] = (result['ATR'] / close) * 100  # ATR as % of price

    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(result)
    result['Stoch_K'] = stoch_k
    result['Stoch_D'] = stoch_d

    # Relative Strength vs SPY (if benchmark provided)
    if benchmark_df is not None:
        result['RS_vs_SPY'] = calculate_relative_strength(close, benchmark_df['Close'])

    # Price change metrics
    result['Pct_Change_1d'] = close.pct_change(1) * 100
    result['Pct_Change_5d'] = close.pct_change(5) * 100
    result['Pct_Change_20d'] = close.pct_change(20) * 100

    # Distance from moving averages (%)
    result['Dist_SMA_20'] = ((close - result['SMA_20']) / result['SMA_20']) * 100
    result['Dist_SMA_50'] = ((close - result['SMA_50']) / result['SMA_50']) * 100
    result['Dist_SMA_200'] = ((close - result['SMA_200']) / result['SMA_200']) * 100

    return result


def get_latest_indicators(df: pd.DataFrame) -> dict:
    """
    Extract the most recent indicator values

    Args:
        df: DataFrame with all indicators calculated

    Returns:
        Dictionary of latest indicator values
    """
    latest = df.iloc[-1]

    return {
        'date': latest.get('Date', df.index[-1]),
        'close': latest['Close'],
        'volume': latest['Volume'],

        # Momentum
        'rsi': latest.get('RSI'),
        'macd': latest.get('MACD'),
        'macd_signal': latest.get('MACD_Signal'),
        'macd_hist': latest.get('MACD_Hist'),
        'stoch_k': latest.get('Stoch_K'),
        'stoch_d': latest.get('Stoch_D'),

        # Trend
        'sma_20': latest.get('SMA_20'),
        'sma_50': latest.get('SMA_50'),
        'sma_200': latest.get('SMA_200'),
        'dist_sma_20': latest.get('Dist_SMA_20'),
        'dist_sma_50': latest.get('Dist_SMA_50'),
        'dist_sma_200': latest.get('Dist_SMA_200'),

        # Volume
        'vol_ratio': latest.get('Vol_Ratio'),
        'vol_sma_20': latest.get('Vol_SMA_20'),

        # Volatility
        'atr': latest.get('ATR'),
        'atr_pct': latest.get('ATR_Pct'),
        'bb_pct': latest.get('BB_Pct'),

        # Performance
        'pct_1d': latest.get('Pct_Change_1d'),
        'pct_5d': latest.get('Pct_Change_5d'),
        'pct_20d': latest.get('Pct_Change_20d'),

        # Relative strength
        'rs_vs_spy': latest.get('RS_vs_SPY'),
    }


if __name__ == "__main__":
    # Test the indicators
    import yfinance as yf

    print("=== Testing Technical Indicators ===\n")

    # Fetch test data
    ticker = 'AAPL'
    df = yf.download(ticker, period="6mo", progress=False)
    df = df.reset_index()

    # Fetch benchmark
    spy = yf.download('SPY', period="6mo", progress=False)
    spy = spy.reset_index()

    # Calculate all indicators
    df_with_indicators = add_all_indicators(df, spy)

    print(f"\n{ticker} Latest Values:")
    latest = get_latest_indicators(df_with_indicators)
    for key, value in latest.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
