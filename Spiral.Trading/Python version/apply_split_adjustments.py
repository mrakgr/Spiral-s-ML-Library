import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("Split Adjustment Pipeline")
print("=" * 60)

# Load split data
splits_file = Path("data/splits/splits_data.csv")
if not splits_file.exists():
    print(f"ERROR: Split data file not found: {splits_file}")
    print("Run download_split_data_optimized.py first")
    exit(1)

print(f"Loading split data from {splits_file}...")
splits_df = pd.read_csv(splits_file)
splits_df['execution_date'] = pd.to_datetime(splits_df['execution_date'])
print(f"  Loaded {len(splits_df)} splits for {splits_df['ticker'].nunique()} tickers")

# Create adjustment factor lookup
# For each ticker, build cumulative adjustment factors over time
print("\nBuilding cumulative adjustment factors...")
adjustment_factors = {}

for ticker in splits_df['ticker'].unique():
    ticker_splits = splits_df[splits_df['ticker'] == ticker].sort_values('execution_date')
    
    # Build cumulative adjustment factor going backwards in time
    # After a split, historical prices need to be divided by the split ratio
    # Example: 2-for-1 split means historical prices should be divided by 2
    adjustments = []
    for _, row in ticker_splits.iterrows():
        adjustments.append({
            'date': row['execution_date'],
            'ratio': row['split_ratio']
        })
    
    adjustment_factors[ticker] = adjustments

print(f"  Built adjustment factors for {len(adjustment_factors)} tickers")

# Function to apply split adjustments
def apply_split_adjustment(df, ticker, date, price_columns=['open', 'high', 'low', 'close']):
    """
    Apply split adjustments to price data.
    All prices BEFORE a split need to be divided by the split ratio.
    """
    if ticker not in adjustment_factors:
        return df
    
    date = pd.to_datetime(date)
    
    # Find all splits that happened AFTER this date
    # (these splits require us to adjust this historical price)
    cumulative_factor = 1.0
    for split in adjustment_factors[ticker]:
        if split['date'] > date:
            cumulative_factor *= split['ratio']
    
    if cumulative_factor != 1.0:
        # Adjust prices
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col] / cumulative_factor
        
        # Adjust volume (reverse - multiply by split ratio)
        if 'volume' in df.columns:
            df['volume'] = df['volume'] * cumulative_factor
    
    return df

# Test on a single file
print("\nTesting adjustment on a sample file...")
test_file = Path("data/daily_aggregates/2021-01-04.csv.gz")
df = pd.read_csv(test_file)
date = pd.to_datetime("2021-01-04")

# Find a ticker with splits to test
test_ticker = None
for ticker in df['ticker'].unique():
    if ticker in adjustment_factors:
        test_ticker = ticker
        break

if test_ticker:
    print(f"  Testing with ticker: {test_ticker}")
    before = df[df['ticker'] == test_ticker].iloc[0]
    print(f"  Before adjustment: close = {before['close']}")
    
    # Apply adjustment to the whole dataframe
    adjusted_rows = []
    for _, row in df.iterrows():
        row_df = pd.DataFrame([row])
        adjusted_row = apply_split_adjustment(row_df, row['ticker'], date)
        adjusted_rows.append(adjusted_row)
    
    adjusted_df = pd.concat(adjusted_rows, ignore_index=True)
    after = adjusted_df[adjusted_df['ticker'] == test_ticker].iloc[0]
    print(f"  After adjustment: close = {after['close']}")
else:
    print("  No tickers with splits found in test file")

print("\n" + "=" * 60)
print("Split adjustment pipeline ready!")
print("\nNext steps:")
print("1. Run this on all files to create adjusted dataset")
print("2. Or integrate adjustment into momentum calculation")
