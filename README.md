# UNLV Spring 2026 Investment Challenge

**Team ASTRYX INVESTING** | Fund: $500,000

Thomas Rucinski, Daniel Toth, Dominik Toth

---

## What This Does

Python-based stock analytics system that scans the entire S&P 500 daily, scores stocks using two quant models, and outputs actionable trade recommendations.

**Two engines work together:**

- **Technical Scanner** - RSI, MACD, Bollinger Bands, relative strength vs SPY. Identifies bullish/bearish setups across 500+ stocks and exports to Excel.
- **Quant Models** - Tracks 24 fundamental "biomarkers" over time, computes 30-day velocity trends, and ranks stocks through a Value Anchor (undervalued quality) and Growth Aggressor (high-momentum) dual-model system.

---

## Quick Start

```bash
pip3 install -r requirements.txt

# Daily pipeline: fetch fundamentals, validate, run quant models
python3 src/main.py pipeline

# Or run steps individually
python3 src/main.py fetch              # Fetch today's S&P 500 fundamentals
python3 src/main.py validate           # Sanity check the data
python3 src/main.py quant              # Run Value + Growth models

# Technical analysis
python3 src/main.py scan               # Full S&P 500 scan → Excel
python3 src/main.py scan --watchlist   # Scan watchlist only
python3 src/main.py chart NVDA META    # Generate charts
python3 src/main.py quick AAPL TSLA    # Quick console scan
python3 src/main.py lookup AAPL        # Detailed fundamentals
```


## Quant Strategy

**70% current fundamentals / 30% historical trend**

| Model | Filter | Ranks By | Detects |
|-------|--------|----------|---------|
| **Value Anchor** | P/E < 20, D/E < 100 | ROE, Current Ratio, Margin | Value compression, efficiency gains |
| **Growth Aggressor** | Rev Growth > 15% | Op Margin, Beta, Short % | Growth acceleration, price momentum |

Daily snapshots are saved to `data/raw_snapshots/` and used to compute 30-day deltas that detect which stocks are *trending* toward value or growth, not just sitting there.

---

## Project Structure

```
src/
├── main.py                # CLI entry point
├── config.py              # Constants and settings
├── data_service.py        # Data fetching with caching
├── data_fetcher.py        # Daily fundamental snapshots
├── historical_backfiller.py # Backfill from 2026-01-01
├── processor.py           # Master merge, imputation, 30-day deltas
├── models.py              # Value Anchor + Growth Aggressor
├── validator.py           # CSV integrity checks
├── indicators.py          # Technical indicators (RSI, MACD, etc.)
├── screener.py            # Bullish/bearish setup detection
├── daily_eval.py          # Daily evaluation → Excel
└── visualizer.py          # Chart generation

run_daily.sh               # Automation script
output/                    # Reports, charts, recommendations
data/raw_snapshots/        # Daily fundamental CSVs (gitignored)
```

---

## Commands

| Command | Description |
|---------|-------------|
| `pipeline` | Full daily pipeline (fetch + validate + quant) |
| `fetch` | Fetch S&P 500 fundamentals to CSV |
| `backfill` | Backfill historical CSVs from 2026-01-01 |
| `validate` | Check CSV integrity |
| `quant` | Run dual-model scoring engine |
| `scan` | Technical analysis scan → Excel |
| `chart NVDA` | Generate technical chart |
| `lookup AAPL` | Detailed fundamental data |
| `quick AAPL` | Quick console scan |
