# UNLV SPRING 2026 INVESTMENT CHALLENGE

**Team: ASTRYX INVESTING**

**Repo Author:** Thomas Rucinski
**Master's Student at UNLV studying Interdisciplinary Biomedical Engineering**

**Managed Fund:** $500,000
**Partners:** Daniel Toth and Dominik Toth
*(Each partner was allotted one third of the fund at the beginning of the challenge)*

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Daily market scan → Excel report
python src/main.py scan

# Scan your watchlist only
python src/main.py scan --watchlist

# Quick console scan of specific tickers
python src/main.py quick NVDA AAPL META

# Generate technical charts
python src/main.py chart NVDA META --period 1y

# Run strategy backtest
python src/main.py backtest --optimized

# Get stock info and position sizing
python src/main.py info AAPL NVDA
```

---

## Commands

| Command | Description |
|---------|-------------|
| `python src/main.py scan` | Run daily market evaluation and export to Excel |
| `python src/main.py scan --watchlist` | Scan only your watchlist |
| `python src/main.py scan --date 2026-01-27` | Get historical closing prices |
| `python src/main.py quick AAPL NVDA` | Quick console scan |
| `python src/main.py chart NVDA` | Generate technical analysis chart |
| `python src/main.py backtest` | Run strategy backtest |
| `python src/main.py info AAPL` | Get stock info + position sizing |

---

## Competition Rules

- **Initial Capital:** $500,000
- **Max Single Position:** 25% of portfolio
- **Max ETF Position:** 50% of portfolio
- **Minimum Buy Price:** $5
- **Minimum Short Price:** $10
- **Minimum Trades:** 10
- **No Day Trading**

---

## Project Structure

```
src/
├── main.py           # Unified CLI entry point (USE THIS)
├── config.py         # All constants and settings
├── data_service.py   # Data fetching (unified)
├── daily_eval.py     # Daily evaluation → Excel
├── indicators.py     # Technical indicators
├── screener.py       # Stock screening logic
├── position_sizer.py # Position sizing calculator
├── backtester.py     # Strategy backtesting
├── visualizer.py     # Chart generation
└── portfolio_manager.py  # Portfolio tracking

output/
├── daily_eval_*.xlsx # Daily evaluation reports
└── charts/           # Generated charts
```

---

## Configuration

Edit `src/config.py` to customize:

- **WATCHLIST**: Your tracked tickers
- **DEFAULT_PORTFOLIO_VALUE**: Your current portfolio value
- **INDICATORS**: Technical indicator settings
- **RISK**: Risk management parameters
- **SCREENING**: Signal thresholds

---

## Output

Daily scans generate Excel files with:

1. **Summary** - Market overview
2. **Screening Results** - All stocks with signals
3. **Bullish Setups** - Buy candidates
4. **Position Sizes** - Recommended position sizing
5. **Sector Performance** - Sector ETF changes
6. **Decision Helper** - Conviction scoring for LONG/SHORT
7. **Trade Journal** - Track your trades with auto-calculating P&L
8. **Daily Log** - Historical tracking
