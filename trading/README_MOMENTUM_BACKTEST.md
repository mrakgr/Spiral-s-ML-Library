# Momentum Backtesting Data Preparation

## Overview
This pipeline prepares data for backtesting a rotational momentum strategy:
- **Strategy**: Hold top 20 stocks by 6-month momentum
- **Rebalancing**: Every 2 weeks (10 trading days)
- **Data**: 5 years of daily data (2020-2025) from Massive
- **Survivorship bias**: Addressed (Massive includes delisted stocks)
- **Split adjustments**: Applied using Polygon API data
- **Dividend adjustments**: Not applied (minimal impact on momentum rankings)

## Pipeline Scripts

### 1. `download_bulk_daily_data.py` ✓ Complete
Downloads 5 years of daily aggregate data from Massive S3.
- Output: `data/daily_aggregates/*.csv.gz`
- ~9,000+ tickers per day
- Already executed

### 2. `download_split_data_optimized.py` ⏳ Running
Fetches stock split data from Polygon API for all tickers.
- Output: `data/splits/splits_data.csv`
- Currently running in background (~55 minutes total)
- Check progress: `tail -f split_download.log`
- Checkpoint-enabled: Can resume if interrupted

### 3. `prepare_momentum_data.py` ⏸ Ready
Main data preparation script that:
1. Loads all daily price data
2. Applies split adjustments (backward adjustment)
3. Calculates 6-month momentum for each ticker/date
4. Filters for liquidity (min volume: 1M shares) and quality (min price: $5)
5. Ranks stocks by momentum on each date
6. Creates rebalancing schedule (every 2 weeks)

**Output:**
- `data/momentum_prepared/momentum_data.parquet` - Main dataset
- `data/momentum_prepared/rebalance_dates.csv` - Rebalancing schedule

**Key columns in output:**
- `ticker`, `date`, `adj_close`, `adj_volume`
- `momentum_6m` - 6-month return
- `momentum_rank` - Rank by momentum (1 = highest)
- `in_top_n` - Boolean: Is this in top 20?
- `is_rebalance_date` - Boolean: Is this a rebalancing date?

## Data Quality Filters

### Liquidity Filter
- Minimum volume: 1,000,000 shares/day
- Ensures realistic execution assumptions

### Price Filter
- Minimum price: $5
- Avoids penny stocks with unreliable data

### History Filter
- Requires 126 trading days (~6 months) of history
- Ensures momentum calculation is valid

## Running the Pipeline

### Current Status
```bash
# 1. Data download - COMPLETE
# Already have 5 years of data in data/daily_aggregates/

# 2. Split data download - IN PROGRESS
tail -f split_download.log  # Monitor progress

# 3. Prepare momentum data - READY TO RUN
# Wait for splits to finish, then:
python3 prepare_momentum_data.py
```

### If Splits Download Interrupted
The script saves checkpoints, so you can safely restart:
```bash
python3 download_split_data_optimized.py
# Will resume from checkpoint
```

## Split Adjustment Methodology

### Why Backward Adjustment?
Stock splits change the number of shares but not company value. For historical analysis:
- **Example**: 2-for-1 split on 2023-01-01
  - After split: Stock trades at $50
  - Before split: Stock traded at $100 (raw)
  - Adjusted: Historical prices divided by 2 → $50
  
This makes momentum calculations consistent across split dates.

### Cumulative Adjustments
If multiple splits occur:
- Adjustments are compounded going backward in time
- Example: 2-for-1 split, then 3-for-1 split
  - Latest prices: No adjustment (factor = 1.0)
  - Before 2nd split: Divide by 3 (factor = 3.0)
  - Before 1st split: Divide by 2×3 = 6 (factor = 6.0)

## Momentum Calculation

### Formula
```
momentum_6m = (current_price - price_126_days_ago) / price_126_days_ago
```

### Trading Days
- 126 trading days ≈ 6 months
- Accounts for weekends/holidays

### Ranking
- On each date, all stocks ranked by momentum_6m
- Rank 1 = highest momentum (best performer)
- Top 20 ranks selected for portfolio

## Rebalancing Schedule

### Frequency
- Every 10 trading days (≈ 2 weeks)
- Fixed schedule based on data availability

### Portfolio Construction
On each rebalancing date:
1. Rank all eligible stocks by 6-month momentum
2. Select top 20 by rank
3. Equal-weight allocation (5% each)
4. Hold until next rebalancing date

## Next Steps

After data preparation completes:

1. **Build backtesting engine**
   - Simulate portfolio rebalancing
   - Track returns, drawdowns, turnover
   - Calculate performance metrics

2. **Add transaction costs**
   - Model bid-ask spread
   - Include commissions
   - Account for market impact

3. **Risk analysis**
   - Volatility, Sharpe ratio, max drawdown
   - Sector concentration
   - Factor exposures

4. **Sensitivity analysis**
   - Different momentum windows (3m, 9m, 12m)
   - Different rebalancing frequencies
   - Different portfolio sizes (10, 30, 50 stocks)

## File Structure
```
trading/
├── data/
│   ├── daily_aggregates/        # Raw daily data from Massive
│   │   ├── 2020-12-09.csv.gz
│   │   ├── ...
│   │   └── 2025-12-04.csv.gz
│   ├── splits/                  # Split adjustment data
│   │   └── splits_data.csv
│   └── momentum_prepared/       # Prepared data (output)
│       ├── momentum_data.parquet
│       └── rebalance_dates.csv
├── download_bulk_daily_data.py
├── download_split_data_optimized.py
├── prepare_momentum_data.py
└── README_MOMENTUM_BACKTEST.md  # This file
```

## Notes

### Survivorship Bias
✓ Addressed - Massive includes delisted stocks

### Dividend Adjustments
✗ Not applied - Minimal impact on momentum rankings
- Dividends don't change relative momentum order significantly
- Most momentum academic studies use split-adjusted only

### Data Limitations
- No intraday data (daily close only)
- Assumes can trade at closing prices
- No bid-ask spread modeling (yet)
- No market impact modeling (yet)

### Performance Expectations
For reference, academic momentum studies show:
- ~10-15% annual returns (before costs)
- High turnover (~300-500% annually)
- Significant drawdowns during market reversals
- Strong performance in trending markets
