# ASTRYX INVESTING - Codebase Revamp Plan

## Goal
Strip all competition/portfolio rules. Make this a pure stock analytics tool with comprehensive fundamental data, defaulting to the full S&P 500.

---

## Phase 1: Delete Competition Files

Delete these 4 files entirely (they're competition-specific or deprecated):
- `src/position_sizer.py` — competition position sizing
- `src/portfolio_manager.py` — competition compliance tracking
- `src/backtester.py` — strategy backtesting with competition rules
- `src/data_collector.py` — deprecated, nothing imports it
- `src/dashboard.py` — deprecated, main.py replaced it

---

## Phase 2: Clean Up config.py

**Remove:**
- `COMPETITION` dict (initial_capital, max positions, min prices, etc.)
- `DEFAULT_PORTFOLIO_VALUE` ($166,600)
- `POSITION_SIZING` dict (strong/moderate/weak percentages)
- `RISK` dict (stop loss multipliers, max positions, max hold days)
- `BACKTEST` dict (period, position size, exclude setups)
- `SP500_TOP_100` hardcoded list — replace with dynamic fetch

**Keep:**
- `WATCHLIST`, `SECTOR_ETFS`, `INDICATORS`, `SCREENING`, `OUTPUT`, `get_figures_dir()`

**Add:**
- `FUNDAMENTALS` dict — config for which fundamental fields to fetch from yfinance

---

## Phase 3: Revamp data_service.py

**Replace `get_sp500_tickers()`:**
- Scrape Wikipedia's S&P 500 table via `pandas.read_html()`
- Cache the ticker list to `output/.cache/sp500_tickers.json` (refresh weekly)
- Fallback to a hardcoded full 500 list if Wikipedia scrape fails
- Default returns ALL tickers, optional `top_n` param still works

**Replace `fetch_multiple()` with batch download:**
- Use `yf.download(tickers_list, period, group_by='ticker', threads=True)` for OHLCV
- Split into chunks of ~50 tickers to avoid API issues
- Much faster than serial fetching with 0.1s delays

**Expand `get_stock_info()` → `get_fundamentals()`:**
- Fetch all 4 categories of fundamental data from yfinance `.info`:
  - **Valuation:** trailingPE, forwardPE, pegRatio, priceToSalesTrailing12Months, enterpriseToEbitda
  - **Growth:** revenueGrowth, earningsGrowth, earningsQuarterlyGrowth
  - **Profitability:** returnOnEquity, profitMargins, freeCashflow, operatingMargins, grossMargins
  - **Health:** debtToEquity, currentRatio, shortPercentOfFloat, shortRatio
  - **Also keep:** sector, industry, marketCap, beta, dividendYield, name

**Add `fetch_fundamentals_batch()`:**
- Use `concurrent.futures.ThreadPoolExecutor` to fetch `.info` in parallel (8-10 workers)
- Cache results to `output/.cache/fundamentals_YYYYMMDD.json` with daily TTL
- Loading cached fundamentals skips the slow API calls entirely

---

## Phase 4: Update screener.py

- Remove any references to competition imports (position_sizer, etc.) — verify there are none
- Add optional fundamental columns to screening output DataFrame (P/E, market cap, sector) when available
- Keep all 7 bullish + 3 bearish setups unchanged (pure technical analysis)

---

## Phase 5: Rewrite main.py

**Remove entirely:**
- `cmd_backtest` command and its parser
- `cmd_info` command (or rework — see below)
- All `--portfolio` arguments
- Imports of `COMPETITION`, `DEFAULT_PORTFOLIO_VALUE`

**Rework `cmd_scan`:**
- Default `--top` to 500 (all S&P 500) instead of 50
- Remove portfolio/position sizing output from print
- Pass fundamentals data through to Excel report

**Rework `cmd_info` → `cmd_lookup`:**
- Show fundamental data for given tickers (P/E, growth, margins, etc.)
- No position sizing, just pure stock info

**Keep:**
- `cmd_chart` (already updated)
- `cmd_quick` (remove portfolio references)

---

## Phase 6: Rewrite daily_eval.py

**Remove entirely:**
- Sheet 4 "Position Sizes" — competition position sizing
- Sheet 6 "Decision Helper" (lines 382-696) — competition scoring/sizing
- Sheet 7 "Trade Journal" — competition trade tracking
- All `PositionSizer` imports and usage
- All `portfolio_value` references
- Position sizing step from `run_daily_evaluation()`

**Keep and enhance:**
- Sheet 1 "Summary" — remove portfolio lines, add fundamental highlights
- Sheet 2 "Screening Results" — add fundamental columns (P/E, Market Cap, Sector)
- Sheet 3 "Bullish Setups" — keep as-is
- Sheet 5 "Sector Performance" — keep as-is
- Sheet 8 "Daily Log" — keep as-is

**Add new:**
- Sheet "Bearish Setups" — split out from screening (currently only bullish gets its own sheet)
- Sheet "Fundamentals" — detailed fundamental data for all scanned stocks

---

## Phase 7: Clean Up visualizer.py

**Remove:**
- `plot_backtest_results()` — backtester is being deleted
- `plot_portfolio_allocation()` — competition portfolio tracking

**Keep:**
- `plot_stock_analysis()` — core technical chart
- `plot_screening_results()` — screening summary
- `create_dashboard_report()` — batch chart creation

---

## Phase 8: Verify & Test

- Run `python main.py scan --watchlist` (quick test with 12 stocks)
- Run `python main.py chart AAPL NVDA` (chart generation)
- Run `python main.py scan` (full S&P 500 — confirm it works)
- Run `python main.py lookup AAPL` (fundamental data display)
- Verify Excel output has correct sheets, no competition references
- Verify no import errors from deleted modules

---

## Summary of Changes

| File | Action |
|------|--------|
| position_sizer.py | DELETE |
| portfolio_manager.py | DELETE |
| backtester.py | DELETE |
| data_collector.py | DELETE |
| dashboard.py | DELETE |
| config.py | Strip competition, add FUNDAMENTALS config |
| data_service.py | Batch downloads, full S&P 500, parallel fundamentals |
| screener.py | Minor — add fundamental cols to output |
| main.py | Remove backtest/portfolio, add lookup, default 500 |
| daily_eval.py | Remove 3 competition sheets, add fundamentals sheet |
| visualizer.py | Remove backtest/portfolio plots |
| indicators.py | No changes needed |
