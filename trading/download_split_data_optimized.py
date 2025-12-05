import json
import requests
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load API credentials
with open("api_key.json") as f:
    api_keys = json.load(f)

# Polygon owns Massive, so we can use the massive_api_key
polygon_api_key = api_keys.get("polygon_api_key") or api_keys.get("massive_api_key")

if not polygon_api_key:
    print("ERROR: polygon_api_key or massive_api_key not found in api_key.json")
    exit(1)

# Create output directory
output_dir = Path("data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

# Output file
output_file = output_dir / "splits_data.csv"

# Load the complete ticker list
ticker_file = Path("data/all_tickers.txt")
if not ticker_file.exists():
    print("ERROR: Ticker list not found. Run get_all_tickers.py first")
    exit(1)

print("Loading complete ticker list...")
with open(ticker_file, 'r') as f:
    ticker_list = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(ticker_list)} unique tickers")

# Date range for split data
start_date = "2020-12-09"
end_date = datetime.now().strftime("%Y-%m-%d")

print(f"\nFetching split data from {start_date} to {end_date}")
print(f"Using {20} concurrent threads (unlimited API plan)")
print("=" * 60)

# Thread-safe counters
lock = threading.Lock()
all_splits = []
failed_tickers = []
progress_count = 0
total_tickers = len(ticker_list)

def fetch_splits_for_ticker(ticker):
    """Fetch split data for a single ticker"""
    url = f"https://api.polygon.io/v3/reference/splits"
    params = {
        "ticker": ticker,
        "execution_date.gte": start_date,
        "execution_date.lte": end_date,
        "order": "asc",
        "limit": 1000,
        "apiKey": polygon_api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        splits = []
        if data.get("results"):
            for split in data["results"]:
                splits.append({
                    "ticker": ticker,
                    "execution_date": split["execution_date"],
                    "split_from": split["split_from"],
                    "split_to": split["split_to"],
                    "split_ratio": split["split_to"] / split["split_from"]
                })
        
        return ticker, splits, None
    except Exception as e:
        return ticker, None, str(e)

# Process tickers with threading (no rate limiting for unlimited plan)
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(fetch_splits_for_ticker, ticker): ticker for ticker in ticker_list}
    
    for future in as_completed(futures):
        ticker = futures[future]
        ticker_symbol, splits, error = future.result()
        
        with lock:
            progress_count += 1
            
            if error:
                failed_tickers.append(ticker_symbol)
                print(f"[{progress_count}/{total_tickers}] {ticker_symbol}: ✗ Error: {error}")
            else:
                if splits:
                    all_splits.extend(splits)
                    print(f"[{progress_count}/{total_tickers}] {ticker_symbol}: ✓ ({len(splits)} split(s))")
                else:
                    print(f"[{progress_count}/{total_tickers}] {ticker_symbol}: ✓ (no splits)")
                
                # Save splits periodically (every 500 tickers)
                if progress_count % 500 == 0 and all_splits:
                    temp_df = pd.DataFrame(all_splits)
                    temp_df.to_csv(output_file, index=False)
                    print(f"  → Checkpoint saved: {len(all_splits)} splits so far")

print("\n" + "=" * 60)
print(f"Split data collection complete!")
print(f"  Total splits found: {len(all_splits)}")
print(f"  Failed tickers: {len(failed_tickers)}")

if all_splits:
    # Save final results
    splits_df = pd.DataFrame(all_splits)
    splits_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved splits data to: {output_file}")
    
    # Show summary
    print(f"\nSplit summary:")
    print(f"  Unique tickers with splits: {splits_df['ticker'].nunique()}")
    print(f"  Date range: {splits_df['execution_date'].min()} to {splits_df['execution_date'].max()}")
    print(f"\nSample of splits:")
    print(splits_df.head(10))
else:
    print("\n⚠ No splits found in the data range")

if failed_tickers:
    print(f"\n⚠ Failed to fetch splits for {len(failed_tickers)} tickers")
