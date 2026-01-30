"""
Portfolio Manager for UNLV Investment Challenge
Enforces competition rules and tracks compliance
"""

import pandas as pd
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class AssetType(Enum):
    STOCK = "stock"
    BOND = "bond"
    MUTUAL_FUND = "mutual_fund"
    ETF = "etf"
    FUTURES = "futures"
    OPTIONS = "options"
    CRYPTO = "crypto"


@dataclass
class Position:
    """Represents a portfolio position"""
    ticker: str
    asset_type: AssetType
    shares: float
    avg_cost: float
    current_price: float
    is_short: bool = False

    @property
    def market_value(self) -> float:
        if self.is_short:
            return -self.shares * self.current_price
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        if self.is_short:
            return (self.avg_cost - self.current_price) * self.shares
        return (self.current_price - self.avg_cost) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0
        return (self.unrealized_pnl / abs(self.cost_basis)) * 100


@dataclass
class Trade:
    """Represents a trade for the trade journal"""
    date: datetime
    ticker: str
    asset_type: AssetType
    action: str  # BUY, SELL, SHORT, COVER
    shares: float
    price: float
    rationale: str  # Required for Trade Journal
    technical_setup: Optional[str] = None
    risk_management: Optional[str] = None

    @property
    def value(self) -> float:
        return self.shares * self.price


# Competition Rules Constants
INITIAL_CAPITAL = 500_000
MAX_SINGLE_SECURITY_PCT = 25  # Max 25% in any single security
MAX_DIVERSIFIED_FUND_PCT = 50  # Max 50% in any single diversified mutual fund/ETF
MIN_BUY_PRICE = 5.0  # Minimum price for buying
MIN_SHORT_PRICE = 10.0  # Minimum price for shorting
MIN_TRADES = 10  # Minimum trades required
REQUIRED_TRADING_MONTHS = ['February', 'March', 'April']


class PortfolioManager:
    """
    Manages portfolio with competition rule enforcement

    Rules enforced:
    - Max 25% in single security
    - Max 50% in single diversified fund/ETF
    - Min $5 buy price, $10 short price
    - No day trading
    - Minimum 10 trades
    - Must trade in Feb, Mar, Apr
    """

    def __init__(self, team_name: str = "ASTRYX INVESTING",
                 num_partners: int = 3):
        self.team_name = team_name
        self.initial_capital = INITIAL_CAPITAL
        self.cash = INITIAL_CAPITAL
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.num_partners = num_partners
        self.capital_per_partner = INITIAL_CAPITAL / num_partners

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value including cash"""
        position_value = sum(
            abs(p.market_value) for p in self.positions.values()
        )
        return self.cash + position_value

    @property
    def total_return(self) -> float:
        """Portfolio return percentage"""
        return ((self.portfolio_value - self.initial_capital) /
                self.initial_capital) * 100

    def check_position_limit(self, ticker: str, value: float,
                            is_diversified_fund: bool = False) -> tuple:
        """
        Check if a position would exceed concentration limits

        Returns:
            (is_allowed, message, max_allowed_value)
        """
        max_pct = MAX_DIVERSIFIED_FUND_PCT if is_diversified_fund else MAX_SINGLE_SECURITY_PCT
        max_value = self.portfolio_value * (max_pct / 100)

        current_value = 0
        if ticker in self.positions:
            current_value = abs(self.positions[ticker].market_value)

        total_value = current_value + value

        if total_value > max_value:
            return (
                False,
                f"{ticker}: ${value:,.0f} would exceed {max_pct}% limit "
                f"(max ${max_value:,.0f}, current ${current_value:,.0f})",
                max_value - current_value
            )

        return (True, "Within limits", max_value - current_value)

    def check_price_limit(self, price: float, is_short: bool = False) -> tuple:
        """
        Check minimum price requirements

        Returns:
            (is_allowed, message)
        """
        if is_short and price < MIN_SHORT_PRICE:
            return (False, f"Short price ${price:.2f} below minimum ${MIN_SHORT_PRICE}")
        if not is_short and price < MIN_BUY_PRICE:
            return (False, f"Buy price ${price:.2f} below minimum ${MIN_BUY_PRICE}")
        return (True, "Price acceptable")

    def check_day_trade(self, ticker: str, action: str) -> tuple:
        """
        Check for day trading violation

        Returns:
            (is_allowed, message)
        """
        today = date.today()

        # Look for opposite action on same day
        opposite_actions = {
            'BUY': ['SELL', 'SHORT'],
            'SELL': ['BUY'],
            'SHORT': ['BUY', 'COVER'],
            'COVER': ['SHORT']
        }

        for trade in self.trades:
            if (trade.date.date() == today and
                trade.ticker == ticker and
                trade.action in opposite_actions.get(action, [])):
                return (
                    False,
                    f"Day trading not allowed: {action} {ticker} after "
                    f"{trade.action} on same day"
                )

        return (True, "No day trade violation")

    def validate_trade(self, ticker: str, shares: float, price: float,
                      action: str, is_diversified_fund: bool = False) -> Dict:
        """
        Validate a proposed trade against all rules

        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'trade_value': shares * price
        }

        is_short = action in ['SHORT']

        # Check price limits
        price_ok, price_msg = self.check_price_limit(price, is_short)
        if not price_ok:
            results['is_valid'] = False
            results['errors'].append(price_msg)

        # Check position limits (for buys/shorts)
        if action in ['BUY', 'SHORT']:
            pos_ok, pos_msg, max_allowed = self.check_position_limit(
                ticker, shares * price, is_diversified_fund
            )
            if not pos_ok:
                results['is_valid'] = False
                results['errors'].append(pos_msg)
            else:
                # Add warning if approaching limit
                pct_of_max = (shares * price) / (self.portfolio_value *
                             (MAX_DIVERSIFIED_FUND_PCT if is_diversified_fund
                              else MAX_SINGLE_SECURITY_PCT) / 100)
                if pct_of_max > 0.8:
                    results['warnings'].append(
                        f"Position is {pct_of_max*100:.0f}% of maximum allowed"
                    )

        # Check day trading
        day_ok, day_msg = self.check_day_trade(ticker, action)
        if not day_ok:
            results['is_valid'] = False
            results['errors'].append(day_msg)

        return results

    def get_compliance_status(self) -> Dict:
        """
        Get current compliance status with all rules

        Returns:
            Dictionary with compliance details
        """
        status = {
            'is_compliant': True,
            'issues': [],
            'warnings': [],
            'trade_count': len(self.trades),
            'trades_needed': max(0, MIN_TRADES - len(self.trades)),
            'monthly_trades': {},
            'position_concentrations': []
        }

        # Check minimum trades
        if len(self.trades) < MIN_TRADES:
            status['warnings'].append(
                f"Only {len(self.trades)}/{MIN_TRADES} minimum trades completed"
            )

        # Check monthly trading requirement
        for trade in self.trades:
            month = trade.date.strftime('%B')
            status['monthly_trades'][month] = status['monthly_trades'].get(month, 0) + 1

        for month in REQUIRED_TRADING_MONTHS:
            if month not in status['monthly_trades']:
                status['warnings'].append(f"No trades yet in {month}")

        # Check position concentrations
        for ticker, pos in self.positions.items():
            pct = abs(pos.market_value) / self.portfolio_value * 100
            is_fund = pos.asset_type in [AssetType.ETF, AssetType.MUTUAL_FUND]
            max_pct = MAX_DIVERSIFIED_FUND_PCT if is_fund else MAX_SINGLE_SECURITY_PCT

            status['position_concentrations'].append({
                'ticker': ticker,
                'value': abs(pos.market_value),
                'pct_of_portfolio': pct,
                'max_allowed_pct': max_pct,
                'headroom': max_pct - pct
            })

            if pct > max_pct:
                status['is_compliant'] = False
                status['issues'].append(
                    f"{ticker} at {pct:.1f}% exceeds {max_pct}% limit"
                )
            elif pct > max_pct * 0.9:
                status['warnings'].append(
                    f"{ticker} at {pct:.1f}% approaching {max_pct}% limit"
                )

        return status

    def generate_trade_journal_entry(self, trade: Trade) -> str:
        """
        Generate formatted trade journal entry for StockTrak

        Required: document reasons for each trade
        """
        entry = f"""
=== TRADE JOURNAL ENTRY ===
Date: {trade.date.strftime('%Y-%m-%d %H:%M')}
Action: {trade.action} {trade.shares:.0f} shares of {trade.ticker}
Price: ${trade.price:.2f}
Value: ${trade.value:,.2f}

RATIONALE:
{trade.rationale}

TECHNICAL SETUP:
{trade.technical_setup or 'N/A'}

RISK MANAGEMENT:
{trade.risk_management or 'N/A'}
==============================
"""
        return entry

    def calculate_position_size(self, price: float, atr: float,
                               risk_pct: float = 1.0,
                               is_diversified_fund: bool = False) -> Dict:
        """
        Calculate optimal position size within rules

        Args:
            price: Current stock price
            atr: Average True Range (14-day)
            risk_pct: Percentage of portfolio to risk
            is_diversified_fund: If True, uses 50% max instead of 25%

        Returns:
            Dictionary with sizing details
        """
        max_position_pct = MAX_DIVERSIFIED_FUND_PCT if is_diversified_fund else MAX_SINGLE_SECURITY_PCT
        max_position_value = self.portfolio_value * (max_position_pct / 100)

        # Risk-based sizing (2x ATR stop)
        risk_per_share = atr * 2
        max_risk = self.portfolio_value * (risk_pct / 100)
        shares_by_risk = int(max_risk / risk_per_share)

        # Position limit sizing
        shares_by_limit = int(max_position_value / price)

        # Take minimum
        shares = min(shares_by_risk, shares_by_limit)
        position_value = shares * price

        return {
            'shares': shares,
            'position_value': position_value,
            'pct_of_portfolio': (position_value / self.portfolio_value) * 100,
            'max_allowed_pct': max_position_pct,
            'stop_loss': price - (atr * 2),
            'risk_amount': shares * risk_per_share,
            'risk_pct': (shares * risk_per_share / self.portfolio_value) * 100,
            'limited_by': 'risk' if shares_by_risk < shares_by_limit else 'position_limit'
        }


def print_portfolio_summary(pm: PortfolioManager):
    """Print formatted portfolio summary"""

    print("\n" + "=" * 70)
    print(f"{pm.team_name} - PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"Initial Capital:    ${pm.initial_capital:>15,.2f}")
    print(f"Current Value:      ${pm.portfolio_value:>15,.2f}")
    print(f"Cash:               ${pm.cash:>15,.2f}")
    print(f"Total Return:       {pm.total_return:>15.2f}%")
    print(f"Capital/Partner:    ${pm.capital_per_partner:>15,.2f}")
    print("-" * 70)

    # Compliance status
    status = pm.get_compliance_status()
    print("\nCOMPLIANCE STATUS:")
    print(f"  Trades Completed: {status['trade_count']}/{MIN_TRADES} minimum")
    print(f"  Monthly Trading: {status['monthly_trades']}")

    if status['issues']:
        print("\n  ⚠️  ISSUES:")
        for issue in status['issues']:
            print(f"    - {issue}")

    if status['warnings']:
        print("\n  ⚡ WARNINGS:")
        for warning in status['warnings']:
            print(f"    - {warning}")

    # Position details
    if pm.positions:
        print("\nPOSITIONS:")
        print("-" * 70)
        print(f"{'Ticker':<8} {'Shares':>10} {'Avg Cost':>10} {'Price':>10} "
              f"{'Value':>12} {'P&L':>10} {'%Port':>8}")
        print("-" * 70)

        for ticker, pos in pm.positions.items():
            print(f"{ticker:<8} {pos.shares:>10.0f} ${pos.avg_cost:>9.2f} "
                  f"${pos.current_price:>9.2f} ${pos.market_value:>11,.0f} "
                  f"{pos.unrealized_pnl_pct:>+9.1f}% "
                  f"{abs(pos.market_value)/pm.portfolio_value*100:>7.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    # Test the portfolio manager
    print("\n=== Portfolio Manager Test ===")

    pm = PortfolioManager(team_name="ASTRYX INVESTING", num_partners=3)

    print(f"\nInitial Setup:")
    print(f"  Portfolio Value: ${pm.portfolio_value:,.2f}")
    print(f"  Cash: ${pm.cash:,.2f}")
    print(f"  Capital per Partner: ${pm.capital_per_partner:,.2f}")

    # Test position sizing
    print("\nPosition Sizing Example (AAPL at $185, ATR $3.50):")
    sizing = pm.calculate_position_size(price=185, atr=3.50, risk_pct=1.0)
    for key, value in sizing.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Test validation
    print("\nTrade Validation Example:")
    validation = pm.validate_trade(
        ticker='AAPL',
        shares=500,
        price=185,
        action='BUY'
    )
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Trade Value: ${validation['trade_value']:,.2f}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")

    # Test low price validation
    print("\nPenny Stock Validation ($3 stock):")
    validation = pm.validate_trade(
        ticker='PENNY',
        shares=1000,
        price=3.0,
        action='BUY'
    )
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Errors: {validation['errors']}")
