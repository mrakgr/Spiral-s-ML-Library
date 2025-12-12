# Trading Gap Detection Issue & Solution

## Problem Discovered

Stocks like **STRC** and **WW** showing extreme momentum (>20,000%) due to **trading gaps**:

### STRC Example
- Last trading: **April 5, 2024** at $2.46
- Trading resumed: **July 30, 2025** at $94.50  
- **Gap: 481 days (16 months)**
- Calculated momentum: **20,184%** ❌

This is **not genuine momentum** - it's a delisting/relisting event.

## Root Cause

The current momentum calculation:
```python
combined_df['price_lag126'] = combined_df.groupby('ticker')['adj_close'].shift(126)
combined_df['momentum_6m'] = (combined_df['adj_close'] - combined_df['price_lag126']) / combined_df['price_lag126']
```

When we shift 126 rows for a stock with a trading gap:
- **Expected**: Compare today to ~126 trading days ago (~180 calendar days)
- **Actual**: STRC on Nov 25, 2025 compares to Feb 5, 2024 (294 days - includes the gap!)

## Solution Implemented (In Code, Pending Execution)

Add gap detection by checking the date difference:

```python
# Shift both price and date
combined_df['price_lag126'] = combined_df.groupby('ticker')['adj_close'].shift(126)
combined_df['date_lag126'] = combined_df.groupby('ticker')['date'].shift(126)

# Calculate actual calendar days between dates
days_since_lag = (combined_df['date'] - combined_df['date_lag126']).dt.days

# Invalidate momentum if gap > 200 days (expected ~180 for 126 trading days)
has_valid_history = days_since_lag <= 200
combined_df.loc[~has_valid_history, 'momentum_6m'] = None
```

This filters out:
- Delistings/relistings
- Extended trading halts
- IPOs with insufficient history
- Ticker symbol changes
- Corporate restructurings

## Files Modified

1. **prepare_momentum_data.py** - Updated with gap detection logic
2. **add_gap_detection.py** - Standalone script to add gap check to existing data

## Memory Issue

The dataset (13.5M rows, 966 MB) exceeds available system memory:
- Total RAM: 15 GB
- Available: ~6.5 GB
- Swap: Full (4 GB used)

**The scripts are correctly written but can't execute due to memory constraints.**

## Recommendations

### Option 1: Run on Larger Machine
Execute `add_gap_detection.py` on a machine with 32+ GB RAM

### Option 2: Process in Chunks
Modify script to process one month of data at a time

### Option 3: Use Existing Data (Current State)
The current dataset works but includes false momentum signals from trading gaps.

**For initial backtesting:** You can proceed with current data but be aware:
- Some extreme momentum stocks (>1000%) are likely gap-related
- Filter out momentum > 500% as a temporary measure
- Or manually exclude known problematic tickers (STRC, WW, etc.)

## Verification After Fix

Once gap detection runs successfully, verify:

```python
strc = df[df['ticker'] == 'STRC']
latest = strc.iloc[-1]
print(latest['momentum_6m'])  # Should be NaN or None
print(latest['is_eligible'])   # Should be False
```

## Impact

**Before fix:**
- Top 20 momentum: Dominated by gap-related false signals
- STRC rank: #1 with 20,184% momentum

**After fix:**
- Trading gaps invalidated
- Top 20 reflects genuine momentum only
- More realistic backtest results

## Next Steps

1. ✅ Identified issue (trading gaps)
2. ✅ Designed solution (date difference check)  
3. ✅ Updated code files
4. ⏸ Pending: Execute on machine with sufficient RAM
5. ⏸ Pending: Regenerate top 20 list
6. ⏸ Pending: Begin backtesting with clean data

---

**Status**: Solution ready, execution blocked by memory constraints
