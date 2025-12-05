"""
Prepare data for momentum backtesting

This script:
1. Loads daily price data
2. Applies split adjustments
3. Calculates 6-month momentum
4. Filters for liquidity/data quality
5. Creates rebalancing dates (every 2 weeks)
6. Outputs a dataset ready for backtesting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MOMENTUM BACKTESTING DATA PREPARATION")
print("=" * 70)

# Configuration
MOMENTUM_DAYS = 126  # ~6 months of trading days
MIN_DOLLAR_VOLUME = 1_000_000  # Minimum monthly average dollar volume
MIN_AVG_PRICE = 5.0  # Minimum 6-month average price to avoid penny stocks
REBALANCE_DAYS = 10  # Rebalance every 2 weeks (10 trading days)
TOP_N = 20  # Hold top 20 stocks by momentum
PRICE_AVG_DAYS = 126  # 6 months for average price calculation
VOLUME_AVG_DAYS = 21  # ~1 month for average dollar volume

data_dir = Path("data/daily_aggregates")
splits_file = Path("data/splits/splits_data.csv")
output_dir = Path("data/momentum_prepared")
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Load split data
print("\n[1/6] Loading split adjustment data...")
if not splits_file.exists():
    print("  ⚠ Warning: No split data found. Proceeding without split adjustments.")
    print("  Run download_split_data_optimized.py to get split data.")
    splits_df = pd.DataFrame()
    adjustment_factors = {}
else:
    splits_df = pd.read_csv(splits_file)
    splits_df['execution_date'] = pd.to_datetime(splits_df['execution_date'])
    print(f"  ✓ Loaded {len(splits_df)} splits for {splits_df['ticker'].nunique()} tickers")
    
    # Build adjustment factors
    adjustment_factors = {}
    for ticker in splits_df['ticker'].unique():
        ticker_splits = splits_df[splits_df['ticker'] == ticker].sort_values('execution_date')
        adjustments = []
        for _, row in ticker_splits.iterrows():
            adjustments.append({
                'date': row['execution_date'],
                'ratio': row['split_ratio']
            })
        adjustment_factors[ticker] = adjustments

# Step 2: Load and combine all daily data
print("\n[2/6] Loading daily price data...")
all_files = sorted(list(data_dir.glob("*.csv.gz")))
print(f"  Found {len(all_files)} files from {all_files[0].name} to {all_files[-1].name}")

# Load all data
print("  Loading all data (this may take a minute)...")
dfs = []
for file in all_files:
    df = pd.read_csv(file)
    # Extract date from filename
    date_str = file.stem.replace('.csv', '')
    df['date'] = pd.to_datetime(date_str)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
print(f"  ✓ Loaded {len(combined_df):,} rows, {combined_df['ticker'].nunique():,} unique tickers")

# Step 3: Apply split adjustments
print("\n[3/6] Applying split adjustments...")
def get_adjustment_factor(ticker, date):
    """Get cumulative adjustment factor for a ticker on a given date"""
    if ticker not in adjustment_factors:
        return 1.0
    
    date = pd.to_datetime(date)
    cumulative_factor = 1.0
    for split in adjustment_factors[ticker]:
        if split['date'] > date:
            cumulative_factor *= split['ratio']
    return cumulative_factor

# Apply adjustments
if adjustment_factors:
    combined_df['adj_factor'] = combined_df.apply(
        lambda row: get_adjustment_factor(row['ticker'], row['date']), axis=1
    )
    combined_df['adj_open'] = combined_df['open'] / combined_df['adj_factor']
    combined_df['adj_high'] = combined_df['high'] / combined_df['adj_factor']
    combined_df['adj_low'] = combined_df['low'] / combined_df['adj_factor']
    combined_df['adj_close'] = combined_df['close'] / combined_df['adj_factor']
    combined_df['adj_volume'] = combined_df['volume'] * combined_df['adj_factor']
    print(f"  ✓ Applied split adjustments")
else:
    # No adjustments available
    combined_df['adj_close'] = combined_df['close']
    combined_df['adj_volume'] = combined_df['volume']
    print(f"  ⚠ No split adjustments applied")

# Step 4: Calculate 6-month momentum and filtering criteria (point-in-time)
print("\n[4/6] Calculating 6-month momentum and filtering criteria...")
combined_df = combined_df.sort_values(['ticker', 'date'])

# Calculate dollar volume
combined_df['dollar_volume'] = combined_df['adj_close'] * combined_df['adj_volume']

# Calculate momentum: (current_price - price_126_days_ago) / price_126_days_ago
combined_df['price_lag126'] = combined_df.groupby('ticker')['adj_close'].shift(MOMENTUM_DAYS)
combined_df['momentum_6m'] = (combined_df['adj_close'] - combined_df['price_lag126']) / combined_df['price_lag126']

# Calculate 6-month average price (using rolling window, point-in-time)
combined_df['avg_price_6m'] = combined_df.groupby('ticker')['adj_close'].transform(
    lambda x: x.rolling(window=PRICE_AVG_DAYS, min_periods=PRICE_AVG_DAYS).mean()
)

# Calculate 1-month average dollar volume (using rolling window, point-in-time)
combined_df['avg_dollar_volume_1m'] = combined_df.groupby('ticker')['dollar_volume'].transform(
    lambda x: x.rolling(window=VOLUME_AVG_DAYS, min_periods=VOLUME_AVG_DAYS).mean()
)

# Mark eligible stocks (those that pass filters at each point in time)
combined_df['is_eligible'] = (
    (combined_df['avg_dollar_volume_1m'] >= MIN_DOLLAR_VOLUME) &
    (combined_df['avg_price_6m'] >= MIN_AVG_PRICE) &
    (combined_df['momentum_6m'].notna())  # Must have momentum calculated
)

print(f"  ✓ Calculated momentum and filters for {len(combined_df):,} rows")

# Step 5: Check eligibility statistics
print("\n[5/6] Eligibility statistics...")
print(f"  Total rows: {len(combined_df):,}")

eligible_df = combined_df[combined_df['is_eligible']].copy()
print(f"  Eligible rows (pass filters): {len(eligible_df):,}")
print(f"  Unique tickers ever eligible: {eligible_df['ticker'].nunique():,}")

# Calculate eligibility rate per date
eligibility_by_date = combined_df.groupby('date').agg({
    'is_eligible': 'sum',
    'ticker': 'count'
}).rename(columns={'is_eligible': 'eligible_count', 'ticker': 'total_count'})
eligibility_by_date['eligibility_rate'] = eligibility_by_date['eligible_count'] / eligibility_by_date['total_count']
print(f"  Average eligible stocks per day: {eligibility_by_date['eligible_count'].mean():.0f}")
print(f"  Average eligibility rate: {eligibility_by_date['eligibility_rate'].mean():.1%}")

# Keep all data but mark eligibility
filtered_df = combined_df.copy()

# Step 6: Create rebalancing schedule and rankings
print("\n[6/6] Creating rebalancing schedule and rankings...")

# Get rebalancing dates (every 2 weeks / 10 trading days)
all_dates = sorted(filtered_df['date'].unique())
# Start after we have enough history for all calculations
min_required_days = max(MOMENTUM_DAYS, PRICE_AVG_DAYS, VOLUME_AVG_DAYS)
rebalance_dates = [all_dates[i] for i in range(min_required_days, len(all_dates), REBALANCE_DAYS)]
print(f"  ✓ Created {len(rebalance_dates)} rebalancing dates")
print(f"  First rebalance: {rebalance_dates[0].date()}")
print(f"  Last rebalance: {rebalance_dates[-1].date()}")

# Rank stocks by momentum on each date - ONLY among eligible stocks
# Non-eligible stocks get NaN rank
filtered_df['momentum_rank'] = filtered_df.groupby('date').apply(
    lambda x: x[x['is_eligible']]['momentum_6m'].rank(ascending=False, method='first')
).reset_index(level=0, drop=True)

# Mark top N stocks (only eligible stocks can be in top N)
filtered_df['in_top_n'] = (filtered_df['momentum_rank'] <= TOP_N) & (filtered_df['is_eligible'])

# Add rebalancing indicator
filtered_df['is_rebalance_date'] = filtered_df['date'].isin(rebalance_dates)

# Save prepared data
output_file = output_dir / "momentum_data.parquet"
filtered_df.to_parquet(output_file, index=False)
print(f"\n✓ Saved prepared data to: {output_file}")
print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

# Save rebalancing dates separately
rebalance_df = pd.DataFrame({'rebalance_date': rebalance_dates})
rebalance_file = output_dir / "rebalance_dates.csv"
rebalance_df.to_csv(rebalance_file, index=False)
print(f"✓ Saved rebalancing dates to: {rebalance_file}")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Date range: {filtered_df['date'].min().date()} to {filtered_df['date'].max().date()}")
print(f"Total trading days: {filtered_df['date'].nunique():,}")
print(f"Unique tickers: {filtered_df['ticker'].nunique():,}")
print(f"Total observations: {len(filtered_df):,}")
print(f"Rebalancing dates: {len(rebalance_dates)}")
print(f"Top {TOP_N} stocks selected on each rebalance date")
print(f"\nFiltering criteria:")
print(f"  Min 6-month avg price: ${MIN_AVG_PRICE}")
print(f"  Min monthly avg dollar volume: ${MIN_DOLLAR_VOLUME:,}")

# Show sample of top momentum stocks on last rebalance date
print(f"\nSample: Top {TOP_N} momentum stocks on {rebalance_dates[-1].date()}:")
last_rebalance = filtered_df[(filtered_df['date'] == rebalance_dates[-1]) & (filtered_df['is_eligible'])]
top_stocks = last_rebalance.nsmallest(TOP_N, 'momentum_rank')[
    ['ticker', 'adj_close', 'momentum_6m', 'avg_price_6m', 'avg_dollar_volume_1m']
]
print(top_stocks.to_string(index=False))

print("\n" + "=" * 70)
print("✓ Data preparation complete!")
print("  Ready for backtesting momentum strategy")
print("=" * 70)
