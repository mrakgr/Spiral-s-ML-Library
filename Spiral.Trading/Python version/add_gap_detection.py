"""
Add trading gap detection to existing momentum dataset

This adds a check to invalidate momentum for stocks with large trading gaps.
"""

import pandas as pd
from pathlib import Path

print("=" * 70)
print("ADDING TRADING GAP DETECTION")
print("=" * 70)

# Load existing dataset
input_file = Path("data/momentum_prepared/momentum_data.parquet")
print(f"\nLoading existing dataset from {input_file}...")
df = pd.read_parquet(input_file)
print(f"Loaded {len(df):,} rows")

# Sort by ticker and date
df = df.sort_values(['ticker', 'date'])

# Add lagged date column
print("\nCalculating trading gap detection...")
df['date_lag126'] = df.groupby('ticker')['date'].shift(126)

# Calculate days since lag
days_since_lag = (df['date'] - df['date_lag126']).dt.days

# Mark rows with valid history (<=200 days)
has_valid_history = days_since_lag <= 200

# Count invalidations
invalid_count = (~has_valid_history & df['momentum_6m'].notna()).sum()
print(f"Found {invalid_count:,} rows with trading gaps > 200 days")

# Invalidate momentum for rows with gaps
df.loc[~has_valid_history, 'momentum_6m'] = None

# Recalculate eligibility
print("\nRecalculating eligibility...")
df['is_eligible'] = (
    (df['avg_dollar_volume_1m'] >= 1_000_000) &
    (df['avg_price_6m'] >= 5.0) &
    (df['momentum_6m'].notna())
)

# Drop temporary column
df.drop(columns=['date_lag126'], inplace=True)

# Recalculate rankings
print("\nRecalculating momentum rankings...")
df['momentum_rank'] = df.groupby('date').apply(
    lambda x: x[x['is_eligible']]['momentum_6m'].rank(ascending=False, method='first')
).reset_index(level=0, drop=True)

# Recalculate top_n
df['in_top_n'] = (df['momentum_rank'] <= 20) & (df['is_eligible'])

# Save updated dataset
output_file = Path("data/momentum_prepared/momentum_data.parquet")
print(f"\nSaving updated dataset to {output_file}...")
df.to_parquet(output_file, index=False)

print(f"✓ Saved {len(df):,} rows")
print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

# Show sample
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

# Check STRC
strc = df[df['ticker'] == 'STRC'].sort_values('date')
latest = strc.iloc[-1]
print(f"\nSTRC latest ({latest['date'].date()}):")
print(f"  Momentum: {latest['momentum_6m']}" if pd.notna(latest['momentum_6m']) else "  Momentum: NaN (invalidated)")
print(f"  Is eligible: {latest['is_eligible']}")
print(f"  Rank: {latest['momentum_rank']}" if pd.notna(latest['momentum_rank']) else "  Rank: NaN")

print("\n" + "=" * 70)
print("✓ Gap detection added successfully!")
print("=" * 70)
