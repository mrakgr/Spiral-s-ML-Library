import gzip
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Input directory with downloaded files
input_dir = Path("data/daily_aggregates")
output_file = input_dir.parent / "combined_daily_data.csv"

if not input_dir.exists():
    print(f"Error: Directory {input_dir} does not exist")
    print("Please run download_bulk_data.py first")
    exit(1)

# Get all .csv.gz files
files = sorted(input_dir.glob("*.csv.gz"))

if not files:
    print(f"No .csv.gz files found in {input_dir}")
    exit(1)

print("="*60)
print(f"Combining {len(files)} daily files into single CSV")
print(f"Input directory: {input_dir.absolute()}")
print(f"Output file: {output_file.absolute()}")
print("="*60)

proceed = input("\nProceed? (y/n): ").lower()
if proceed != 'y':
    print("Aborted.")
    exit(0)

print("\nReading and combining files...")
all_data = []
failed_files = []

for file_path in tqdm(files, desc="Processing files"):
    try:
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f)
            all_data.append(df)
    except Exception as e:
        print(f"\nError reading {file_path.name}: {e}")
        failed_files.append(file_path.name)

if not all_data:
    print("Error: No data was successfully read from files")
    exit(1)

print("\nConcatenating DataFrames...")
combined_df = pd.concat(all_data, ignore_index=True)

print("Saving to CSV...")
combined_df.to_csv(output_file, index=False)

# Print summary
print("\n" + "="*60)
print("✓ Combined data saved successfully!")
print("="*60)
print(f"Output file: {output_file.absolute()}")
print(f"File size: {output_file.stat().st_size / (1024**2):.1f} MB")
print(f"\nData Summary:")
print(f"  Total rows: {len(combined_df):,}")
print(f"  Columns: {list(combined_df.columns)}")

if 'ticker' in combined_df.columns:
    print(f"  Unique tickers: {combined_df['ticker'].nunique():,}")
    
if 'timestamp' in combined_df.columns:
    combined_df['date'] = pd.to_datetime(combined_df['timestamp'], unit='ms')
    print(f"  Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
elif any(col in combined_df.columns for col in ['date', 'time', 'datetime']):
    date_col = next(col for col in ['date', 'time', 'datetime'] if col in combined_df.columns)
    print(f"  Date range: {combined_df[date_col].min()} to {combined_df[date_col].max()}")

if failed_files:
    print(f"\n⚠ Warning: {len(failed_files)} files failed to read:")
    for f in failed_files[:10]:
        print(f"  - {f}")
    if len(failed_files) > 10:
        print(f"  ... and {len(failed_files) - 10} more")

print("\n✓ First few rows:")
print(combined_df.head(10))

print("\n✓ Done!")
