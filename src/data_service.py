"""
ASTRYX INVESTING - Unified Data Service
Single source of truth for all data fetching operations
"""

import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import SECTOR_ETFS, FUNDAMENTALS

# Cache directory for persistent caching between runs
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', '.cache')


class DataService:
    """
    Centralized data fetching service.
    Uses batch yfinance downloads and parallel fundamentals fetching.
    """

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}
        self._spy_cache: Optional[pd.DataFrame] = None

    def clear_cache(self):
        self._cache.clear()
        self._spy_cache = None

    # =========================================================================
    # CORE DATA FETCHING
    # =========================================================================

    def fetch_stock(self, ticker: str, period: str = "1y",
                    interval: str = "1d", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Fetch historical data for a single ticker."""
        cache_key = f"{ticker}_{period}_{interval}"

        if use_cache and self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                return None

            df = self._standardize_dataframe(df, ticker)

            if self.cache_enabled:
                self._cache[cache_key] = df.copy()

            return df

        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            return None

    def fetch_multiple(self, tickers: List[str], period: str = "1y",
                       interval: str = "1d",
                       show_progress: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers using yfinance batch download.
        Much faster than serial fetching — downloads all tickers in parallel.
        """
        if not tickers:
            return {}

        data = {}
        failed = []

        # Use yf.download for batch fetching (much faster than serial)
        chunk_size = 50
        chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        if show_progress:
            print(f"  Downloading {len(tickers)} tickers in {len(chunks)} batch(es)...")

        for chunk_idx, chunk in enumerate(chunks):
            try:
                batch_df = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    group_by='ticker',
                    threads=True,
                    progress=show_progress and chunk_idx == 0,
                )

                if batch_df.empty:
                    failed.extend(chunk)
                    continue

                # Single ticker returns flat columns, multiple returns MultiIndex
                if len(chunk) == 1:
                    ticker = chunk[0]
                    df = batch_df.copy()
                    df = self._standardize_dataframe(df, ticker)
                    if not df.empty:
                        data[ticker] = df
                        if self.cache_enabled:
                            self._cache[f"{ticker}_{period}_{interval}"] = df.copy()
                    else:
                        failed.append(ticker)
                else:
                    for ticker in chunk:
                        try:
                            if ticker in batch_df.columns.get_level_values(0):
                                ticker_df = batch_df[ticker].copy()
                                ticker_df = ticker_df.dropna(how='all')
                                if not ticker_df.empty:
                                    ticker_df = self._standardize_dataframe(ticker_df, ticker)
                                    data[ticker] = ticker_df
                                    if self.cache_enabled:
                                        self._cache[f"{ticker}_{period}_{interval}"] = ticker_df.copy()
                                else:
                                    failed.append(ticker)
                            else:
                                failed.append(ticker)
                        except Exception:
                            failed.append(ticker)

            except Exception as e:
                print(f"  Batch download error: {e}")
                failed.extend(chunk)

        if failed and show_progress:
            print(f"  Failed ({len(failed)}): {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

        return data

    def fetch_spy(self, period: str = "1y", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Fetch SPY benchmark data."""
        if use_cache and self._spy_cache is not None:
            return self._spy_cache.copy()

        df = self.fetch_stock('SPY', period, use_cache=False)

        if df is not None and self.cache_enabled:
            self._spy_cache = df.copy()

        return df

    def fetch_sectors(self, period: str = "5d") -> Dict[str, pd.DataFrame]:
        """Fetch sector ETF data."""
        return self.fetch_multiple(
            list(SECTOR_ETFS.keys()),
            period=period,
            show_progress=False
        )

    # =========================================================================
    # DATE FILTERING
    # =========================================================================

    def filter_to_date(self, df: pd.DataFrame,
                       target_date: Union[str, date, datetime]) -> pd.DataFrame:
        """Filter dataframe to include only data up to and including target date."""
        if df is None or df.empty:
            return df

        if isinstance(target_date, str):
            target_dt = pd.to_datetime(target_date).date()
        elif isinstance(target_date, datetime):
            target_dt = target_date.date()
        else:
            target_dt = target_date

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
        """Get the closing price for a specific date (or most recent)."""
        if df is None or df.empty:
            return 0.0

        if target_date:
            df = self.filter_to_date(df, target_date)

        close_col = df['Close']

        if hasattr(close_col.iloc[-1], 'iloc'):
            return float(close_col.iloc[-1].iloc[0])
        return float(close_col.iloc[-1])

    # =========================================================================
    # FUNDAMENTALS
    # =========================================================================

    def get_fundamentals(self, ticker: str) -> dict:
        """
        Get comprehensive fundamental data for a single ticker.
        Includes valuation, growth, profitability, and health metrics.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            result = {'ticker': ticker}

            # Pull all configured fundamental fields
            for category, fields in FUNDAMENTALS.items():
                for field in fields:
                    result[field] = info.get(field)

            # Derive a clean name
            result['name'] = info.get('longName', info.get('shortName', ticker))

            return result

        except Exception:
            result = {'ticker': ticker, 'name': ticker}
            for category, fields in FUNDAMENTALS.items():
                for field in fields:
                    result[field] = None
            return result

    def fetch_fundamentals_batch(self, tickers: List[str],
                                 max_workers: int = 8,
                                 show_progress: bool = True,
                                 use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch fundamentals for multiple tickers in parallel.
        Caches results to disk daily to avoid repeated slow API calls.
        """
        # Check disk cache first
        if use_cache:
            cached = self._load_fundamentals_cache(tickers)
            if cached is not None:
                return cached

        results = []
        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_fundamentals, t): t for t in tickers}

            if show_progress:
                pbar = tqdm(total=len(tickers), desc="Fundamentals")

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception:
                    failed.append(ticker)
                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        if failed:
            print(f"  Failed fundamentals ({len(failed)}): {', '.join(failed[:10])}")

        df = pd.DataFrame(results)

        # Save to disk cache
        if use_cache and not df.empty:
            self._save_fundamentals_cache(df)

        return df

    def _load_fundamentals_cache(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """Load cached fundamentals from disk if fresh (same day)."""
        cache_file = os.path.join(
            _CACHE_DIR, f"fundamentals_{datetime.now().strftime('%Y%m%d')}.json"
        )
        if not os.path.exists(cache_file):
            return None

        try:
            df = pd.read_json(cache_file)
            # Check if cache has all requested tickers
            cached_tickers = set(df['ticker'].tolist())
            requested = set(tickers)
            if requested.issubset(cached_tickers):
                print("  Using cached fundamentals (today's data)")
                return df[df['ticker'].isin(requested)].reset_index(drop=True)
            return None
        except Exception:
            return None

    def _save_fundamentals_cache(self, df: pd.DataFrame):
        """Save fundamentals to disk cache."""
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(
            _CACHE_DIR, f"fundamentals_{datetime.now().strftime('%Y%m%d')}.json"
        )

        # Merge with existing cache if present
        if os.path.exists(cache_file):
            try:
                existing = pd.read_json(cache_file)
                # Update existing entries, add new ones
                combined = pd.concat([existing, df]).drop_duplicates(
                    subset='ticker', keep='last'
                ).reset_index(drop=True)
                combined.to_json(cache_file, orient='records', indent=2)
                return
            except Exception:
                pass

        df.to_json(cache_file, orient='records', indent=2)

    # =========================================================================
    # S&P 500 TICKER LIST
    # =========================================================================

    @staticmethod
    def get_sp500_tickers(top_n: Optional[int] = None) -> List[str]:
        """
        Get S&P 500 tickers by scraping Wikipedia.
        Caches the list locally for a week to avoid repeated scraping.
        Returns all ~500 tickers by default.
        """
        cache_file = os.path.join(_CACHE_DIR, 'sp500_tickers.json')
        os.makedirs(_CACHE_DIR, exist_ok=True)

        # Check cache (refresh weekly)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                cache_date = datetime.strptime(cached['date'], '%Y-%m-%d')
                if (datetime.now() - cache_date).days < 7:
                    tickers = cached['tickers']
                    return tickers[:top_n] if top_n else tickers
            except Exception:
                pass

        # Scrape Wikipedia
        try:
            tables = pd.read_html(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            )
            df = tables[0]
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()

            # Save cache
            with open(cache_file, 'w') as f:
                json.dump({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'tickers': tickers,
                }, f)

            print(f"  Loaded {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers[:top_n] if top_n else tickers

        except Exception as e:
            print(f"  Wikipedia scrape failed ({e}), using fallback list")
            return _SP500_FALLBACK[:top_n] if top_n else _SP500_FALLBACK

    @staticmethod
    def get_sector_etfs() -> Dict[str, str]:
        return SECTOR_ETFS.copy()

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _standardize_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = df.reset_index()
        df['Ticker'] = ticker

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df.columns = [col.replace(' ', '_') for col in df.columns]
        return df

    def flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_default_service = None


def _get_service() -> DataService:
    global _default_service
    if _default_service is None:
        _default_service = DataService()
    return _default_service


def get_sp500_tickers(top_n: Optional[int] = None) -> List[str]:
    return DataService.get_sp500_tickers(top_n)


def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    return _get_service().fetch_stock(ticker, period, interval)


def fetch_bulk_data(tickers: List[str], period: str = "1y",
                    interval: str = "1d") -> Dict[str, pd.DataFrame]:
    return _get_service().fetch_multiple(tickers, period, interval)


def fetch_spy_benchmark(period: str = "1y") -> Optional[pd.DataFrame]:
    return _get_service().fetch_spy(period)


def get_stock_info(ticker: str) -> dict:
    return _get_service().get_fundamentals(ticker)


def fetch_stock_metadata(tickers: List[str]) -> pd.DataFrame:
    return _get_service().fetch_fundamentals_batch(tickers)


# =============================================================================
# FALLBACK S&P 500 LIST (if Wikipedia scrape fails)
# =============================================================================
_SP500_FALLBACK = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE',
    'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK',
    'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN',
    'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO',
    'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX',
    'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR',
    'BLK', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG',
    'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CDNS',
    'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF',
    'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP',
    'COF', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CRC', 'CRL', 'CRM',
    'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX',
    'CZR', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR',
    'DIS', 'DISH', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK',
    'DVA', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX',
    'EL', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES',
    'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR',
    'F', 'FANG', 'FAST', 'FBHS', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FIS',
    'FISV', 'FITB', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRC', 'FRT', 'FTNT', 'FTV',
    'GD', 'GE', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL',
    'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD',
    'PEAK', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL',
    'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF',
    'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM',
    'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JKHY', 'JNJ', 'JNPR',
    'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI',
    'KMX', 'KO', 'KR', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ',
    'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUMN', 'LUV', 'LVS', 'LW',
    'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO',
    'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC',
    'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO',
    'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ',
    'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC',
    'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'NXPI', 'O',
    'ODFL', 'OGN', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PARA',
    'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG',
    'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW',
    'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PVH', 'PWR', 'PXD',
    'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI', 'RJF',
    'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'SBAC', 'SBNY',
    'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO',
    'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF',
    'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC',
    'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV',
    'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR',
    'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VFC', 'VICI',
    'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB',
    'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB',
    'WMT', 'WRB', 'WRK', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XRAY',
    'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS',
]


# =============================================================================
# CLI TEST
# =============================================================================
if __name__ == "__main__":
    print("=== ASTRYX INVESTING - Data Service Test ===\n")

    service = DataService()

    # Test S&P 500 ticker list
    print("Testing S&P 500 ticker list...")
    tickers = service.get_sp500_tickers()
    print(f"  Got {len(tickers)} tickers")

    # Test single fetch
    print("\nTesting single stock fetch...")
    aapl = service.fetch_stock('AAPL', period='6mo')
    if aapl is not None:
        print(f"  AAPL: {len(aapl)} days")
        print(f"  Latest close: ${service.get_closing_price(aapl):.2f}")

    # Test batch fetch
    print("\nTesting batch fetch (5 tickers)...")
    data = service.fetch_multiple(['NVDA', 'META', 'TSLA', 'AAPL', 'MSFT'], period='3mo')
    for ticker, df in data.items():
        print(f"  {ticker}: {len(df)} days")

    # Test fundamentals
    print("\nTesting fundamentals fetch...")
    fund = service.get_fundamentals('AAPL')
    print(f"  AAPL P/E: {fund.get('trailingPE')}")
    print(f"  AAPL ROE: {fund.get('returnOnEquity')}")
    print(f"  AAPL Debt/Equity: {fund.get('debtToEquity')}")

    print("\n Done!")
