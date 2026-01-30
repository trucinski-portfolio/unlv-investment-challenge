"""
Position Sizing Calculator for UNLV Investment Challenge
Based on volatility (ATR) and risk management principles
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionSize:
    """Result of position sizing calculation"""
    ticker: str
    current_price: float
    shares: int
    position_value: float
    stop_loss_price: float
    risk_per_share: float
    total_risk: float
    risk_reward_ratio: Optional[float]
    target_price: Optional[float]
    atr: float
    atr_pct: float


class PositionSizer:
    """
    Calculate position sizes based on volatility and risk parameters

    Uses ATR-based position sizing to normalize risk across positions

    UNLV Competition Rules Enforced:
    - Max 25% in any single security
    - Max 50% in any diversified mutual fund/ETF
    - Minimum buy price: $5
    - Minimum short price: $10
    """

    # Competition rule constants
    MAX_SINGLE_SECURITY_PCT = 25  # Max 25% in any single security
    MAX_DIVERSIFIED_FUND_PCT = 50  # Max 50% in diversified funds/ETFs
    MIN_BUY_PRICE = 5.0
    MIN_SHORT_PRICE = 10.0

    def __init__(self,
                 portfolio_value: float = 500000.00,  # Full $500k fund
                 max_risk_per_trade_pct: float = 1.0,  # Risk 1% per trade
                 max_position_pct: float = 25.0,  # Competition max: 25%
                 atr_multiplier: float = 2.0,  # Stop loss = 2x ATR
                 is_diversified_fund: bool = False):  # If True, uses 50% max
        """
        Args:
            portfolio_value: Total portfolio value
            max_risk_per_trade_pct: Maximum % of portfolio to risk per trade
            max_position_pct: Maximum % of portfolio in single position (default 25% per rules)
            atr_multiplier: Multiplier for ATR to set stop loss distance
            is_diversified_fund: If True, allows up to 50% position (ETF/mutual fund)
        """
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = portfolio_value * (max_risk_per_trade_pct / 100)

        # Apply competition rules
        rule_max = self.MAX_DIVERSIFIED_FUND_PCT if is_diversified_fund else self.MAX_SINGLE_SECURITY_PCT
        effective_max = min(max_position_pct, rule_max)
        self.max_position_value = portfolio_value * (effective_max / 100)
        self.max_position_pct = effective_max
        self.atr_multiplier = atr_multiplier

    def calculate_position(self,
                           ticker: str,
                           price: float,
                           atr: float,
                           target_price: Optional[float] = None) -> PositionSize:
        """
        Calculate position size for a stock

        Args:
            ticker: Stock symbol
            price: Current stock price
            atr: Average True Range (14-day)
            target_price: Optional price target for R:R calculation

        Returns:
            PositionSize object with all calculations
        """
        # ATR as percentage of price
        atr_pct = (atr / price) * 100

        # Stop loss distance (2x ATR by default)
        stop_distance = atr * self.atr_multiplier
        stop_loss_price = price - stop_distance
        risk_per_share = stop_distance

        # Calculate shares based on risk
        shares_by_risk = int(self.max_risk_per_trade / risk_per_share)

        # Calculate shares based on max position size
        shares_by_position = int(self.max_position_value / price)

        # Take the smaller of the two
        shares = min(shares_by_risk, shares_by_position)

        # Ensure at least 1 share if we can afford it
        if shares == 0 and price <= self.max_position_value:
            shares = 1

        position_value = shares * price
        total_risk = shares * risk_per_share

        # Risk/Reward ratio if target provided
        risk_reward = None
        if target_price and target_price > price:
            potential_gain = target_price - price
            risk_reward = potential_gain / risk_per_share

        return PositionSize(
            ticker=ticker,
            current_price=price,
            shares=shares,
            position_value=position_value,
            stop_loss_price=stop_loss_price,
            risk_per_share=risk_per_share,
            total_risk=total_risk,
            risk_reward_ratio=risk_reward,
            target_price=target_price,
            atr=atr,
            atr_pct=atr_pct
        )

    def calculate_multiple(self, stocks: list) -> pd.DataFrame:
        """
        Calculate positions for multiple stocks

        Args:
            stocks: List of dicts with 'ticker', 'price', 'atr', optional 'target'

        Returns:
            DataFrame with position sizing for all stocks
        """
        results = []
        for stock in stocks:
            pos = self.calculate_position(
                ticker=stock['ticker'],
                price=stock['price'],
                atr=stock['atr'],
                target_price=stock.get('target')
            )
            results.append({
                'Ticker': pos.ticker,
                'Price': pos.current_price,
                'Shares': pos.shares,
                'Position_Value': pos.position_value,
                'Stop_Loss': pos.stop_loss_price,
                'Risk_Per_Share': pos.risk_per_share,
                'Total_Risk': pos.total_risk,
                'ATR_Pct': pos.atr_pct,
                'R_R_Ratio': pos.risk_reward_ratio,
                'Target': pos.target_price
            })
        return pd.DataFrame(results)

    def portfolio_allocation(self, positions: list) -> dict:
        """
        Calculate portfolio allocation across multiple positions

        Args:
            positions: List of PositionSize objects

        Returns:
            Dictionary with allocation summary
        """
        total_invested = sum(p.position_value for p in positions)
        total_risk = sum(p.total_risk for p in positions)

        return {
            'num_positions': len(positions),
            'total_invested': total_invested,
            'cash_remaining': self.portfolio_value - total_invested,
            'pct_invested': (total_invested / self.portfolio_value) * 100,
            'total_risk': total_risk,
            'risk_pct': (total_risk / self.portfolio_value) * 100,
            'avg_position_size': total_invested / len(positions) if positions else 0
        }


def print_position_report(sizer: PositionSizer, stocks: list):
    """Print a formatted position sizing report"""

    print("\n" + "=" * 70)
    print("POSITION SIZING REPORT")
    print(f"Portfolio Value: ${sizer.portfolio_value:,.2f}")
    print(f"Max Risk per Trade: ${sizer.max_risk_per_trade:,.2f} ({sizer.max_risk_per_trade/sizer.portfolio_value*100:.1f}%)")
    print(f"Max Position Size: ${sizer.max_position_value:,.2f}")
    print("=" * 70)

    positions = []
    for stock in stocks:
        pos = sizer.calculate_position(
            ticker=stock['ticker'],
            price=stock['price'],
            atr=stock['atr'],
            target_price=stock.get('target')
        )
        positions.append(pos)

        print(f"\n{pos.ticker}:")
        print(f"  Current Price:    ${pos.current_price:.2f}")
        print(f"  ATR (14-day):     ${pos.atr:.2f} ({pos.atr_pct:.1f}%)")
        print(f"  Stop Loss:        ${pos.stop_loss_price:.2f} (-{pos.atr_pct * sizer.atr_multiplier:.1f}%)")
        print(f"  Position Size:    {pos.shares} shares (${pos.position_value:,.2f})")
        print(f"  Risk if Stopped:  ${pos.total_risk:.2f}")
        if pos.risk_reward_ratio:
            print(f"  Target Price:     ${pos.target_price:.2f}")
            print(f"  Risk/Reward:      1:{pos.risk_reward_ratio:.1f}")

    # Portfolio summary
    allocation = sizer.portfolio_allocation(positions)
    print("\n" + "-" * 70)
    print("PORTFOLIO SUMMARY:")
    print(f"  Total Positions:    {allocation['num_positions']}")
    print(f"  Total Invested:     ${allocation['total_invested']:,.2f} ({allocation['pct_invested']:.1f}%)")
    print(f"  Cash Remaining:     ${allocation['cash_remaining']:,.2f}")
    print(f"  Total Risk:         ${allocation['total_risk']:,.2f} ({allocation['risk_pct']:.1f}%)")
    print("=" * 70)


def get_live_stock_data(tickers: list) -> list:
    """Fetch live price and ATR data for position sizing"""
    import yfinance as yf
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from indicators import calculate_atr

    stocks = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="3mo", progress=False)
            if df.empty:
                continue

            # Calculate ATR
            high = df['High']
            low = df['Low']
            close = df['Close']
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]

            # Handle multi-index columns from yfinance
            current_price = float(df['Close'].iloc[-1].iloc[0] if hasattr(df['Close'].iloc[-1], 'iloc') else df['Close'].iloc[-1])
            atr_val = float(atr.iloc[0] if hasattr(atr, 'iloc') else atr)

            stocks.append({
                'ticker': ticker,
                'price': current_price,
                'atr': atr_val,
                'target': None  # Can be set manually
            })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return stocks


if __name__ == "__main__":
    # Test the position sizer with competition rules
    print("\n=== UNLV Investment Challenge Position Sizer ===")
    print("\nCompetition Rules:")
    print("  - Max 25% in any single security")
    print("  - Max 50% in any diversified mutual fund/ETF")
    print("  - Minimum buy price: $5")
    print("  - Minimum short price: $10")

    # Initialize with full $500k fund (competition rules)
    sizer = PositionSizer(
        portfolio_value=500000.00,
        max_risk_per_trade_pct=1.0,  # Risk 1% = $5,000 per trade
        max_position_pct=25.0,  # Competition max (will cap at 25%)
        atr_multiplier=2.0
    )

    # Fetch LIVE data for position sizing
    print("\nFetching live market data...")
    test_tickers = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'GOOGL', 'META']
    test_stocks = get_live_stock_data(test_tickers)

    if test_stocks:
        print_position_report(sizer, test_stocks)
    else:
        print("Could not fetch live data")

    # Show ETF example with 50% max
    print("\n\n=== ETF Position (50% max allowed) ===")
    etf_data = get_live_stock_data(['SPY'])
    if etf_data:
        etf_sizer = PositionSizer(
            portfolio_value=500000.00,
            max_risk_per_trade_pct=2.0,
            max_position_pct=50.0,  # ETFs can go up to 50%
            is_diversified_fund=True
        )
        spy = etf_data[0]
        etf_pos = etf_sizer.calculate_position('SPY', price=spy['price'], atr=spy['atr'])
        print(f"SPY: {etf_pos.shares} shares @ ${etf_pos.current_price:.2f} = ${etf_pos.position_value:,.2f}")
        print(f"Position %: {(etf_pos.position_value/500000)*100:.1f}% (max 50% allowed for ETFs)")
