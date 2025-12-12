"""
Create SQLite database from daily aggregate data

This creates a normalized database structure that can handle large datasets efficiently.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("CREATING SQLITE DATABASE FROM DAILY AGGREGATES")
print("=" * 70)

# Database file
db_file = Path("data/trading.db")
if db_file.exists():
    print(f"\n⚠ Database already exists: {db_file}")
    response = input("Delete and recreate? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        exit(0)
    db_file.unlink()
    print("✓ Deleted existing database")

# Connect to database
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create tables
print("\nCreating database schema...")

# Main daily prices table
cursor.execute("""
CREATE TABLE daily_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    transactions INTEGER,
    UNIQUE(ticker, date)
)
""")

# Create indexes for fast lookups
cursor.execute("CREATE INDEX idx_ticker_date ON daily_prices(ticker, date)")
cursor.execute("CREATE INDEX idx_date ON daily_prices(date)")
cursor.execute("CREATE INDEX idx_ticker ON daily_prices(ticker)")

conn.commit()
print("✓ Created daily_prices table with indexes")

# Load data from CSV files
data_dir = Path("data/daily_aggregates")
all_files = sorted(list(data_dir.glob("*.csv.gz")))

print(f"\nFound {len(all_files)} files to load")
print(f"Date range: {all_files[0].stem} to {all_files[-1].stem}")
print("\nLoading data (this will take several minutes)...")

total_rows = 0
batch_size = 100  # Process 100 files at a time
errors = []

for i in range(0, len(all_files), batch_size):
    batch_files = all_files[i:i+batch_size]
    batch_dfs = []
    
    for file_path in batch_files:
        try:
            # Read file
            df = pd.read_csv(file_path)
            
            # Extract date from filename
            date_str = file_path.stem.replace('.csv', '')
            df['date'] = date_str
            
            # Select and rename columns
            df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'transactions']]
            
            # Filter out rows with missing tickers
            df = df[df['ticker'].notna()]
            
            batch_dfs.append(df)
            
        except Exception as e:
            errors.append((file_path.name, str(e)))
            continue
    
    # Combine batch and insert
    if batch_dfs:
        batch_df = pd.concat(batch_dfs, ignore_index=True)
        
        # Insert into database
        batch_df.to_sql('daily_prices', conn, if_exists='append', index=False)
        
        total_rows += len(batch_df)
        
        # Progress
        progress = (i + len(batch_files)) / len(all_files) * 100
        print(f"  Progress: {progress:.1f}% ({i + len(batch_files)}/{len(all_files)} files) - {total_rows:,} rows loaded")

conn.commit()

print("\n" + "=" * 70)
print("DATA LOADING COMPLETE")
print("=" * 70)
print(f"Total rows loaded: {total_rows:,}")

if errors:
    print(f"\n⚠ Errors encountered: {len(errors)}")
    for filename, error in errors[:10]:
        print(f"  {filename}: {error}")

# Verify data
print("\nVerifying database...")
cursor.execute("SELECT COUNT(*) FROM daily_prices")
row_count = cursor.fetchone()[0]
print(f"✓ Database contains {row_count:,} rows")

cursor.execute("SELECT COUNT(DISTINCT ticker) FROM daily_prices")
ticker_count = cursor.fetchone()[0]
print(f"✓ Database contains {ticker_count:,} unique tickers")

cursor.execute("SELECT MIN(date), MAX(date) FROM daily_prices")
min_date, max_date = cursor.fetchone()
print(f"✓ Date range: {min_date} to {max_date}")

# Show database size
db_size_mb = db_file.stat().st_size / 1024 / 1024
print(f"✓ Database size: {db_size_mb:.1f} MB")

# Sample query
print("\nSample query - NVDA last 5 days:")
cursor.execute("""
    SELECT date, ticker, open, high, low, close, volume
    FROM daily_prices
    WHERE ticker = 'NVDA'
    ORDER BY date DESC
    LIMIT 5
""")
for row in cursor.fetchall():
    print(f"  {row}")

conn.close()

print("\n" + "=" * 70)
print("✓ Database created successfully!")
print(f"  Location: {db_file}")
print("=" * 70)
print("\nNext steps:")
print("1. Load splits data into database")
print("2. Create views for split-adjusted prices")
print("3. Calculate momentum using SQL queries")
