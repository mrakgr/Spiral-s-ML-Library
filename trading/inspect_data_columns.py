import gzip
import pandas as pd
from pathlib import Path

# Directory with downloaded daily aggregates
data_dir = Path("trading/data/daily_aggregates")

# Get the first available file
files = sorted(data_dir.glob("*.csv.gz"))

if not files:
    print("No files found")
    exit(1)

print(f"Inspecting first file: {files[0].name}")
print("="*60)

try:
    with gzip.open(files[0], 'rt') as f:
        df = pd.read_csv(f)
        
        print(f"Columns: {list(df.columns)}")
        print(f"\nTotal rows: {len(df)}")
        print(f"\nFirst few rows:")
        print(df.head(10))
        
        # Check if NVDA exists
        if 'ticker' in df.columns:
            nvda = df[df['ticker'] == 'NVDA']
            if not nvda.empty:
                print(f"\n✓ Found NVDA in this file:")
                print(nvda)
            else:
                print("\n! NVDA not found in this file")
        else:
            print("\n! No 'ticker' column found")
            
except Exception as e:
    print(f"Error: {e}")
