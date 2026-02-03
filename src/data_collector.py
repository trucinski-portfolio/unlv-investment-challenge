"""
Data Collection Module for UNLV Investment Challenge

NOTE: This module is maintained for backward compatibility.
Prefer using data_service.py instead:

    from data_service import DataService
    service = DataService()
    df = service.fetch_stock('AAPL', period='1y')
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time


# S&P 500 Top 100 Holdings (by market cap weight as of 2024)
SP500_TOP_100 = [
    # Top 10 (Mega caps)
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'LLY', 'AVGO', 'JPM',
    # 11-20
    'TSLA', 'UNH', 'XOM', 'V', 'MA', 'COST', 'PG', 'JNJ', 'HD', 'WMT',
    # 21-30
    'ABBV', 'NFLX', 'CRM', 'BAC', 'CVX', 'MRK', 'KO', 'AMD', 'PEP', 'TMO',
    # 31-40
    'LIN', 'ORCL', 'ACN', 'CSCO', 'MCD', 'ADBE', 'WFC', 'IBM', 'ABT', 'PM',
    # 41-50
    'GE', 'CAT', 'NOW', 'DIS', 'ISRG', 'QCOM', 'TXN', 'INTU', 'VZ', 'BKNG',
    # 51-60
    'CMCSA', 'AMGN', 'AMAT', 'SPGI', 'NEE', 'PFE', 'T', 'AXP', 'MS', 'LOW',
    # 61-70
    'HON', 'UNP', 'GS', 'RTX', 'BLK', 'UBER', 'ELV', 'SYK', 'VRTX', 'C',
    # 71-80
    'TJX', 'SCHW', 'PANW', 'BSX', 'PGR', 'DE', 'FI', 'LRCX', 'BMY', 'GILD',
    # 81-90
    'MMC', 'CB', 'SBUX', 'ADI', 'ETN', 'KKR', 'MDT', 'MU', 'REGN', 'SO',
    # 91-100
    'KLAC', 'CI', 'MDLZ', 'CME', 'CL', 'ZTS', 'APH', 'ICE', 'DUK', 'EOG'
]

# Sector ETFs for relative strength comparison
SECTOR_ETFS = {
    'SPY': 'S&P 500',
    'QQQ': 'Nasdaq 100',
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XBI': 'Biotech',
    'IBB': 'Biotech (iShares)'
}


def get_sp500_tickers(top_n: int = 100) -> list:
    """Return top N S&P 500 tickers by market cap weight"""
    return SP500_TOP_100[:top_n]


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical data for a single ticker

    Args:
        ticker: Stock symbol
        period: Data period (e.g., "2y" for 2 years)
        interval: Data interval (e.g., "1d" for daily)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return None

        # Clean up the dataframe
        df = df.reset_index()
        df['Ticker'] = ticker

        # Rename columns for consistency
        df.columns = [col.replace(' ', '_') for col in df.columns]

        return df

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def fetch_bulk_data(tickers: list, period: str = "2y", interval: str = "1d",
                    delay: float = 0.1) -> dict:
    """
    Fetch historical data for multiple tickers

    Args:
        tickers: List of stock symbols
        period: Data period
        interval: Data interval
        delay: Delay between requests to avoid rate limiting

    Returns:
        Dictionary of {ticker: DataFrame}
    """
    data = {}
    failed = []

    print(f"\nFetching data for {len(tickers)} tickers...")

    for ticker in tqdm(tickers, desc="Downloading"):
        df = fetch_stock_data(ticker, period, interval)
        if df is not None and not df.empty:
            data[ticker] = df
        else:
            failed.append(ticker)
        time.sleep(delay)  # Be nice to Yahoo Finance

    if failed:
        print(f"\nFailed to fetch: {failed}")

    print(f"Successfully fetched {len(data)} of {len(tickers)} tickers")
    return data


def fetch_spy_benchmark(period: str = "2y") -> pd.DataFrame:
    """Fetch SPY data for relative strength calculations"""
    return fetch_stock_data('SPY', period)


def get_stock_info(ticker: str) -> dict:
    """Get basic stock info (sector, industry, market cap)"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'ticker': ticker,
            'name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'avg_volume': info.get('averageVolume', 0),
            'beta': info.get('beta', 1.0)
        }
    except Exception as e:
        return {
            'ticker': ticker,
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'avg_volume': 0,
            'beta': 1.0
        }


def fetch_stock_metadata(tickers: list) -> pd.DataFrame:
    """Fetch metadata for multiple stocks"""
    metadata = []

    print(f"\nFetching metadata for {len(tickers)} tickers...")

    for ticker in tqdm(tickers, desc="Getting info"):
        info = get_stock_info(ticker)
        metadata.append(info)
        time.sleep(0.05)

    return pd.DataFrame(metadata)


if __name__ == "__main__":
    # Test the data collector
    print("=== UNLV Investment Challenge Data Collector ===\n")

    # Test with a small subset
    test_tickers = ['AAPL', 'MSFT', 'NVDA', 'SPY']

    print("Testing with:", test_tickers)
    data = fetch_bulk_data(test_tickers, period="6mo")

    for ticker, df in data.items():
        print(f"\n{ticker}: {len(df)} days of data")
        print(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(3))
