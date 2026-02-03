"""
ASTRYX INVESTING - Unified Data Service
Single source of truth for all data fetching operations
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import time

from config import SP500_TOP_100, SECTOR_ETFS, INDICATORS


class DataService:
    """
    Centralized data fetching service.
    Use this instead of scattered yfinance calls throughout the codebase.
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the data service.

        Args:
            cache_enabled: Whether to cache fetched data in memory
        """
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}
        self._spy_cache: Optional[pd.DataFrame] = None

    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
        self._spy_cache = None

    # =========================================================================
    # CORE DATA FETCHING
    # =========================================================================

    def fetch_stock(self, ticker: str, period: str = "1y",
                    interval: str = "1d", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single ticker.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Data period ('6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1d', '1wk', '1mo')
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data, or None if fetch fails
        """
        cache_key = f"{ticker}_{period}_{interval}"

        # Check cache
        if use_cache and self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                return None

            # Standardize the dataframe
            df = self._standardize_dataframe(df, ticker)

            # Cache if enabled
            if self.cache_enabled:
                self._cache[cache_key] = df.copy()

            return df

        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            return None

    def fetch_multiple(self, tickers: List[str], period: str = "1y",
                       interval: str = "1d", delay: float = 0.1,
                       show_progress: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers.

        Args:
            tickers: List of stock symbols
            period: Data period
            interval: Data interval
            delay: Delay between requests (to avoid rate limiting)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary of {ticker: DataFrame}
        """
        data = {}
        failed = []

        iterator = tqdm(tickers, desc="Fetching") if show_progress else tickers

        for ticker in iterator:
            df = self.fetch_stock(ticker, period, interval)
            if df is not None and not df.empty:
                data[ticker] = df
            else:
                failed.append(ticker)
            time.sleep(delay)

        if failed and show_progress:
            print(f"  Failed: {', '.join(failed)}")

        return data

    def fetch_spy(self, period: str = "1y", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch SPY benchmark data.

        Args:
            period: Data period
            use_cache: Whether to use cached SPY data

        Returns:
            DataFrame with SPY data
        """
        if use_cache and self._spy_cache is not None:
            return self._spy_cache.copy()

        df = self.fetch_stock('SPY', period, use_cache=False)

        if df is not None and self.cache_enabled:
            self._spy_cache = df.copy()

        return df

    def fetch_sectors(self, period: str = "5d") -> Dict[str, pd.DataFrame]:
        """
        Fetch sector ETF data.

        Args:
            period: Data period

        Returns:
            Dictionary of {etf_symbol: DataFrame}
        """
        return self.fetch_multiple(
            list(SECTOR_ETFS.keys()),
            period=period,
            delay=0.05,
            show_progress=False
        )

    # =========================================================================
    # DATE FILTERING
    # =========================================================================

    def filter_to_date(self, df: pd.DataFrame,
                       target_date: Union[str, date, datetime]) -> pd.DataFrame:
        """
        Filter dataframe to include only data up to and including target date.
        Use this when you need historical closing prices for a specific date.

        Args:
            df: DataFrame with datetime index
            target_date: The date to filter to (string 'YYYY-MM-DD' or date object)

        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty:
            return df

        # Convert target_date to date object
        if isinstance(target_date, str):
            target_dt = pd.to_datetime(target_date).date()
        elif isinstance(target_date, datetime):
            target_dt = target_date.date()
        else:
            target_dt = target_date

        # Get the date column or index
        if 'Date' in df.columns:
            df_dates = pd.to_datetime(df['Date']).dt.date
            mask = df_dates <= target_dt
        else:
            df_dates = pd.to_datetime(df.index).date
            mask = df_dates <= target_dt

        filtered = df[mask]

        if filtered.empty:
            print(f"    Warning: No data found on or before {target_dt}")
            return df

        return filtered

    def get_closing_price(self, df: pd.DataFrame,
                          target_date: Optional[Union[str, date]] = None) -> float:
        """
        Get the closing price for a specific date (or most recent).

        Args:
            df: DataFrame with price data
            target_date: Optional date to get price for

        Returns:
            Closing price as float
        """
        if df is None or df.empty:
            return 0.0

        if target_date:
            df = self.filter_to_date(df, target_date)

        close_col = df['Close']

        # Handle multi-index columns from yfinance
        if hasattr(close_col.iloc[-1], 'iloc'):
            return float(close_col.iloc[-1].iloc[0])
        return float(close_col.iloc[-1])

    # =========================================================================
    # STOCK INFO & METADATA
    # =========================================================================

    def get_stock_info(self, ticker: str) -> dict:
        """
        Get basic stock info (sector, industry, market cap, etc.)

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with stock metadata
        """
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
                'beta': info.get('beta', 1.0),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
            }
        except Exception:
            return {
                'ticker': ticker,
                'name': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'avg_volume': 0,
                'beta': 1.0,
                'pe_ratio': None,
                'dividend_yield': None,
            }

    def fetch_metadata(self, tickers: List[str],
                       show_progress: bool = True) -> pd.DataFrame:
        """
        Fetch metadata for multiple stocks.

        Args:
            tickers: List of stock symbols
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with stock metadata
        """
        metadata = []
        iterator = tqdm(tickers, desc="Getting info") if show_progress else tickers

        for ticker in iterator:
            info = self.get_stock_info(ticker)
            metadata.append(info)
            time.sleep(0.05)

        return pd.DataFrame(metadata)

    # =========================================================================
    # TICKER LISTS
    # =========================================================================

    @staticmethod
    def get_sp500_tickers(top_n: int = 100) -> List[str]:
        """Return top N S&P 500 tickers by market cap weight"""
        return SP500_TOP_100[:top_n]

    @staticmethod
    def get_sector_etfs() -> Dict[str, str]:
        """Return sector ETF mapping"""
        return SECTOR_ETFS.copy()

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _standardize_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardize DataFrame format for consistency"""
        df = df.reset_index()
        df['Ticker'] = ticker

        # Handle multi-index columns from yfinance bulk downloads
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Standardize column names
        df.columns = [col.replace(' ', '_') for col in df.columns]

        return df

    def flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten multi-index columns that yfinance sometimes returns.
        Call this if you're getting weird column issues.
        """
        if df is None:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        return df


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================
# These provide backward compatibility with the old data_collector.py interface

_default_service = None


def _get_service() -> DataService:
    """Get or create the default DataService instance"""
    global _default_service
    if _default_service is None:
        _default_service = DataService()
    return _default_service


def get_sp500_tickers(top_n: int = 100) -> List[str]:
    """Return top N S&P 500 tickers by market cap weight"""
    return DataService.get_sp500_tickers(top_n)


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch historical data for a single ticker"""
    return _get_service().fetch_stock(ticker, period, interval)


def fetch_bulk_data(tickers: List[str], period: str = "2y",
                    interval: str = "1d", delay: float = 0.1) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for multiple tickers"""
    return _get_service().fetch_multiple(tickers, period, interval, delay)


def fetch_spy_benchmark(period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch SPY data for relative strength calculations"""
    return _get_service().fetch_spy(period)


def get_stock_info(ticker: str) -> dict:
    """Get basic stock info"""
    return _get_service().get_stock_info(ticker)


def fetch_stock_metadata(tickers: List[str]) -> pd.DataFrame:
    """Fetch metadata for multiple stocks"""
    return _get_service().fetch_metadata(tickers)


# =============================================================================
# CLI TEST
# =============================================================================
if __name__ == "__main__":
    print("=== ASTRYX INVESTING - Data Service Test ===\n")

    service = DataService()

    # Test single fetch
    print("Testing single stock fetch...")
    aapl = service.fetch_stock('AAPL', period='6mo')
    if aapl is not None:
        print(f"  AAPL: {len(aapl)} days")
        print(f"  Latest close: ${service.get_closing_price(aapl):.2f}")

    # Test SPY benchmark
    print("\nTesting SPY benchmark...")
    spy = service.fetch_spy(period='6mo')
    if spy is not None:
        print(f"  SPY: {len(spy)} days")

    # Test date filtering
    print("\nTesting date filtering...")
    if aapl is not None:
        filtered = service.filter_to_date(aapl, '2025-12-31')
        print(f"  Filtered to 2025-12-31: {len(filtered)} days")
        print(f"  Closing price: ${service.get_closing_price(filtered):.2f}")

    # Test multiple tickers
    print("\nTesting bulk fetch...")
    data = service.fetch_multiple(['NVDA', 'META', 'TSLA'], period='3mo')
    for ticker, df in data.items():
        print(f"  {ticker}: {len(df)} days")

    print("\n Done!")
