"""
Calculate momentum using SQL queries

This uses SQL window functions and CTEs to efficiently calculate:
1. Split-adjusted prices
2. 6-month momentum
3. Trading gap detection
4. Eligibility filtering
5. Momentum rankings
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("CALCULATING MOMENTUM WITH SQL")
print("=" * 70)

# Connect to database
db_file = Path("data/trading.db")
conn = sqlite3.connect(db_file)

# Configuration
MOMENTUM_DAYS = 126  # ~6 months of trading days
MIN_DOLLAR_VOLUME = 1_000_000  # Minimum monthly average dollar volume
MIN_AVG_PRICE = 5.0  # Minimum 6-month average price
MAX_GAP_DAYS = 200  # Maximum calendar days for valid momentum calculation
PRICE_AVG_DAYS = 126  # 6 months for average price
VOLUME_AVG_DAYS = 21  # ~1 month for average dollar volume

print("\nConfiguration:")
print(f"  Momentum lookback: {MOMENTUM_DAYS} trading days")
print(f"  Min avg price: ${MIN_AVG_PRICE}")
print(f"  Min dollar volume: ${MIN_DOLLAR_VOLUME:,}")
print(f"  Max gap days: {MAX_GAP_DAYS}")

# Step 1: Create split-adjusted prices view
print("\n[1/5] Creating split-adjusted prices view...")

conn.execute("DROP VIEW IF EXISTS adjusted_prices")

conn.execute("""
CREATE VIEW adjusted_prices AS
WITH split_factors AS (
    -- Calculate cumulative adjustment factor for each ticker/date combination
    -- For each date, multiply all split ratios that occurred AFTER that date
    SELECT 
        dp.id,
        dp.ticker,
        dp.date,
        dp.open,
        dp.high,
        dp.low,
        dp.close,
        dp.volume,
        COALESCE(
            (SELECT EXP(SUM(LOG(s.split_ratio)))
             FROM splits s
             WHERE s.ticker = dp.ticker
             AND s.execution_date > dp.date),
            1.0
        ) AS adj_factor
    FROM daily_prices dp
)
SELECT
    id,
    ticker,
    date,
    open / adj_factor AS adj_open,
    high / adj_factor AS adj_high,
    low / adj_factor AS adj_low,
    close / adj_factor AS adj_close,
    CAST(volume * adj_factor AS INTEGER) AS adj_volume,
    close * volume AS dollar_volume
FROM split_factors
""")

print("✓ Created adjusted_prices view")

# Step 2: Calculate momentum with gap detection
print("\n[2/5] Creating momentum calculations view...")

conn.execute("DROP VIEW IF EXISTS momentum_calc")

conn.execute(f"""
CREATE VIEW momentum_calc AS
WITH lagged_data AS (
    -- Get lagged prices and dates using window functions
    SELECT
        ticker,
        date,
        adj_close,
        adj_volume,
        dollar_volume,
        LAG(adj_close, {MOMENTUM_DAYS}) OVER (PARTITION BY ticker ORDER BY date) AS price_lag126,
        LAG(date, {MOMENTUM_DAYS}) OVER (PARTITION BY ticker ORDER BY date) AS date_lag126
    FROM adjusted_prices
),
momentum AS (
    -- Calculate momentum and check for trading gaps
    SELECT
        ticker,
        date,
        adj_close,
        adj_volume,
        dollar_volume,
        price_lag126,
        date_lag126,
        JULIANDAY(date) - JULIANDAY(date_lag126) AS days_since_lag,
        CASE
            WHEN price_lag126 IS NOT NULL 
            AND (JULIANDAY(date) - JULIANDAY(date_lag126)) <= {MAX_GAP_DAYS}
            THEN (adj_close - price_lag126) / price_lag126
            ELSE NULL
        END AS momentum_6m
    FROM lagged_data
)
SELECT * FROM momentum
""")

print("✓ Created momentum_calc view")

# Step 3: Calculate rolling averages for filtering
print("\n[3/5] Creating eligibility filters view...")

conn.execute("DROP VIEW IF EXISTS eligibility_filters")

conn.execute(f"""
CREATE VIEW eligibility_filters AS
WITH rolling_windows AS (
    SELECT
        mc.ticker,
        mc.date,
        mc.adj_close,
        mc.momentum_6m,
        -- 6-month average price
        AVG(ap.adj_close) OVER (
            PARTITION BY mc.ticker 
            ORDER BY mc.date 
            ROWS BETWEEN {PRICE_AVG_DAYS - 1} PRECEDING AND CURRENT ROW
        ) AS avg_price_6m,
        -- Count for min_periods check on price
        COUNT(ap.adj_close) OVER (
            PARTITION BY mc.ticker 
            ORDER BY mc.date 
            ROWS BETWEEN {PRICE_AVG_DAYS - 1} PRECEDING AND CURRENT ROW
        ) AS price_count,
        -- 1-month average dollar volume
        AVG(ap.dollar_volume) OVER (
            PARTITION BY mc.ticker 
            ORDER BY mc.date 
            ROWS BETWEEN {VOLUME_AVG_DAYS - 1} PRECEDING AND CURRENT ROW
        ) AS avg_dollar_volume_1m,
        -- Count for min_periods check on volume
        COUNT(ap.dollar_volume) OVER (
            PARTITION BY mc.ticker 
            ORDER BY mc.date 
            ROWS BETWEEN {VOLUME_AVG_DAYS - 1} PRECEDING AND CURRENT ROW
        ) AS volume_count
    FROM momentum_calc mc
    JOIN adjusted_prices ap ON mc.ticker = ap.ticker AND mc.date = ap.date
)
SELECT
    ticker,
    date,
    adj_close,
    momentum_6m,
    CASE WHEN price_count >= {PRICE_AVG_DAYS} THEN avg_price_6m ELSE NULL END AS avg_price_6m,
    CASE WHEN volume_count >= {VOLUME_AVG_DAYS} THEN avg_dollar_volume_1m ELSE NULL END AS avg_dollar_volume_1m,
    CASE
        WHEN momentum_6m IS NOT NULL
        AND price_count >= {PRICE_AVG_DAYS}
        AND volume_count >= {VOLUME_AVG_DAYS}
        AND avg_price_6m >= {MIN_AVG_PRICE}
        AND avg_dollar_volume_1m >= {MIN_DOLLAR_VOLUME}
        THEN 1
        ELSE 0
    END AS is_eligible
FROM rolling_windows
""")

print("✓ Created eligibility_filters view")

# Step 4: Create momentum rankings
print("\n[4/5] Creating momentum rankings view...")

conn.execute("DROP VIEW IF EXISTS momentum_rankings")

conn.execute("""
CREATE VIEW momentum_rankings AS
SELECT
    ticker,
    date,
    adj_close,
    momentum_6m,
    avg_price_6m,
    avg_dollar_volume_1m,
    is_eligible,
    CASE
        WHEN is_eligible = 1
        THEN RANK() OVER (PARTITION BY date ORDER BY momentum_6m DESC)
        ELSE NULL
    END AS momentum_rank
FROM eligibility_filters
""")

print("✓ Created momentum_rankings view")

# Step 5: Create rebalancing dates table
print("\n[5/5] Creating rebalancing dates...")

# Get all unique dates
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT date FROM daily_prices ORDER BY date")
all_dates = [row[0] for row in cursor.fetchall()]

# Calculate rebalancing dates (every 10 trading days, starting after 126 days)
REBALANCE_DAYS = 10
min_required_days = max(MOMENTUM_DAYS, PRICE_AVG_DAYS, VOLUME_AVG_DAYS)
rebalance_dates = [all_dates[i] for i in range(min_required_days, len(all_dates), REBALANCE_DAYS)]

print(f"  Generated {len(rebalance_dates)} rebalancing dates")
print(f"  First: {rebalance_dates[0]}")
print(f"  Last: {rebalance_dates[-1]}")

# Create rebalancing dates table
conn.execute("DROP TABLE IF EXISTS rebalance_dates")
conn.execute("""
CREATE TABLE rebalance_dates (
    date DATE PRIMARY KEY
)
""")

# Insert rebalancing dates
for date in rebalance_dates:
    conn.execute("INSERT INTO rebalance_dates (date) VALUES (?)", (date,))

conn.commit()
print("✓ Created rebalance_dates table")

# Verification
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

# Test query - Get top 20 momentum stocks for latest rebalance date
latest_rebalance = rebalance_dates[-1]
print(f"\nTop 20 momentum stocks on {latest_rebalance}:")

query = """
SELECT 
    ticker,
    ROUND(adj_close, 2) as price,
    ROUND(momentum_6m * 100, 2) as momentum_pct,
    ROUND(avg_price_6m, 2) as avg_6m_price,
    ROUND(avg_dollar_volume_1m, 0) as avg_dollar_vol,
    momentum_rank
FROM momentum_rankings
WHERE date = ?
AND is_eligible = 1
ORDER BY momentum_rank
LIMIT 20
"""

df = pd.read_sql_query(query, conn, params=(latest_rebalance,))
print(df.to_string(index=False))

# Check WOLF specifically
print(f"\nWOLF status on {latest_rebalance}:")
wolf_query = """
SELECT 
    ticker,
    date,
    ROUND(adj_close, 2) as price,
    ROUND(momentum_6m * 100, 2) as momentum_pct,
    ROUND(avg_price_6m, 2) as avg_6m_price,
    ROUND(avg_dollar_volume_1m, 0) as avg_dollar_vol,
    is_eligible,
    momentum_rank
FROM momentum_rankings
WHERE ticker = 'WOLF'
AND date = ?
"""

wolf_df = pd.read_sql_query(wolf_query, conn, params=(latest_rebalance,))
if len(wolf_df) > 0:
    print(wolf_df.to_string(index=False))
else:
    print("WOLF not found for this date")

# Check WOLF's full trading history
print(f"\nWOLF full trading history:")
wolf_history_query = """
SELECT 
    MIN(date) as first_date,
    MAX(date) as last_date,
    COUNT(*) as trading_days,
    JULIANDAY(MAX(date)) - JULIANDAY(MIN(date)) as calendar_days
FROM daily_prices
WHERE ticker = 'WOLF'
"""

wolf_history = pd.read_sql_query(wolf_history_query, conn)
print(wolf_history.to_string(index=False))

# Check recent WOLF data with lag information
print(f"\nWOLF recent data (last 10 rows):")
wolf_recent_query = """
SELECT 
    date,
    ROUND(adj_close, 2) as price,
    date_lag126,
    ROUND(price_lag126, 2) as price_126_ago,
    days_since_lag,
    ROUND(momentum_6m * 100, 2) as momentum_pct
FROM momentum_calc
WHERE ticker = 'WOLF'
ORDER BY date DESC
LIMIT 10
"""

wolf_recent = pd.read_sql_query(wolf_recent_query, conn)
print(wolf_recent.to_string(index=False))

# Database statistics
print("\n" + "=" * 70)
print("DATABASE STATISTICS")
print("=" * 70)

stats = conn.execute("""
SELECT 
    COUNT(DISTINCT date) as trading_days,
    COUNT(DISTINCT ticker) as total_tickers,
    SUM(CASE WHEN is_eligible = 1 THEN 1 ELSE 0 END) as eligible_observations,
    COUNT(*) as total_observations
FROM momentum_rankings
WHERE date >= (SELECT MIN(date) FROM rebalance_dates)
""").fetchone()

print(f"Trading days with momentum data: {stats[0]:,}")
print(f"Total tickers: {stats[1]:,}")
print(f"Eligible observations: {stats[2]:,}")
print(f"Total observations: {stats[3]:,}")
print(f"Eligibility rate: {stats[2]/stats[3]*100:.1f}%")

conn.close()

print("\n" + "=" * 70)
print("✓ Momentum calculation complete!")
print("=" * 70)
print("\nViews created:")
print("  - adjusted_prices: Split-adjusted OHLCV data")
print("  - momentum_calc: 6-month momentum with gap detection")
print("  - eligibility_filters: Filtering criteria applied")
print("  - momentum_rankings: Final rankings for eligible stocks")
print("\nTables created:")
print("  - rebalance_dates: Rebalancing schedule")
print("\nYou can now query the database directly using SQL!")
