import pandas as pd
from pathlib import Path

print("Extracting all unique tickers from the dataset...")
print("=" * 60)

data_dir = Path("data/daily_aggregates")
all_files = sorted(list(data_dir.glob("*.csv.gz")))

if not all_files:
    print("ERROR: No data files found in data/daily_aggregates/")
    exit(1)

print(f"Found {len(all_files)} files to process")
print(f"Date range: {all_files[0].stem} to {all_files[-1].stem}")

all_tickers = set()
file_count = 0

for file_path in all_files:
    file_count += 1
    if file_count % 100 == 0 or file_count == len(all_files):
        print(f"  Processing file {file_count}/{len(all_files)}: {file_path.name}")
    
    df = pd.read_csv(file_path)
    # Filter out NaN tickers and convert to string
    tickers = [str(t) for t in df['ticker'].unique() if pd.notna(t)]
    all_tickers.update(tickers)

print("\n" + "=" * 60)
print(f"Total unique tickers found: {len(all_tickers)}")

# Save to file
output_file = Path("data/all_tickers.txt")
with open(output_file, 'w') as f:
    for ticker in sorted(all_tickers):
        f.write(f"{ticker}\n")

print(f"✓ Saved ticker list to: {output_file}")
print(f"\nSample tickers (first 20):")
for ticker in sorted(all_tickers)[:20]:
    print(f"  {ticker}")
