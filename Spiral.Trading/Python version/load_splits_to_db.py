"""
Load splits data into SQLite database
"""

import sqlite3
import pandas as pd
from pathlib import Path

print("=" * 70)
print("LOADING SPLITS DATA TO DATABASE")
print("=" * 70)

# Connect to database
db_file = Path("data/trading.db")
if not db_file.exists():
    print(f"ERROR: Database not found: {db_file}")
    print("Run create_database.py first")
    exit(1)

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create splits table
print("\nCreating splits table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS splits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    execution_date DATE NOT NULL,
    split_from REAL NOT NULL,
    split_to REAL NOT NULL,
    split_ratio REAL NOT NULL,
    UNIQUE(ticker, execution_date)
)
""")

# Create index
cursor.execute("CREATE INDEX IF NOT EXISTS idx_splits_ticker ON splits(ticker)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_splits_date ON splits(execution_date)")

conn.commit()
print("✓ Created splits table")

# Load splits data
splits_file = Path("data/splits/splits_data.csv")
if not splits_file.exists():
    print(f"\n⚠ No splits data found at {splits_file}")
    print("Database created without splits data")
    conn.close()
    exit(0)

print(f"\nLoading splits from {splits_file}...")
splits_df = pd.read_csv(splits_file)
print(f"Found {len(splits_df)} splits for {splits_df['ticker'].nunique()} tickers")

# Remove duplicates (keep first occurrence)
splits_df = splits_df.drop_duplicates(subset=['ticker', 'execution_date'], keep='first')
print(f"After removing duplicates: {len(splits_df)} splits")

# Insert into database
splits_df.to_sql('splits', conn, if_exists='append', index=False)
conn.commit()

print("✓ Loaded splits data")

# Verify
cursor.execute("SELECT COUNT(*) FROM splits")
split_count = cursor.fetchone()[0]
print(f"✓ Database contains {split_count:,} splits")

cursor.execute("SELECT COUNT(DISTINCT ticker) FROM splits")
ticker_count = cursor.fetchone()[0]
print(f"✓ Splits for {ticker_count:,} unique tickers")

# Sample query
print("\nSample - NVDA splits:")
cursor.execute("""
    SELECT ticker, execution_date, split_from, split_to, split_ratio
    FROM splits
    WHERE ticker = 'NVDA'
    ORDER BY execution_date
""")
for row in cursor.fetchall():
    print(f"  {row}")

conn.close()

print("\n" + "=" * 70)
print("✓ Splits data loaded successfully!")
print("=" * 70)
