"""
Download daily stock data using Massive REST API (works with free Basic tier)
Uses the grouped daily endpoint to get all stocks for each day
Rate limit: 5 calls/minute on Basic tier
"""

import json
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from massive import RESTClient

# Load API credentials
with open("api_key.json") as f:
    api_keys = json.load(f)

# Use the REST API key (not S3 credentials)
api_key = api_keys.get("massive_api_key") or api_keys.get("massive")

if not api_key:
    print("ERROR: API key not found in api_key.json")
    print("Please add 'massive_api_key' or 'massive' to api_key.json")
    exit(1)

client = RESTClient(api_key)

# Calculate date range for past 2 years
# Free tier has 2 years of historical data
end_date = datetime.now() - timedelta(days=1)  # Yesterday (today's data may not be ready)
start_date = end_date - timedelta(days=730)  # 2 years

# Create output directory
output_dir = Path("data/daily_api_data")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Massive API Bulk Downloader (Free Tier Compatible)")
print("="*60)
print(f"Date range: {start_date.date()} to {end_date.date()}")
print(f"Output directory: {output_dir.absolute()}")
print(f"\nNOTE: Free tier has 5 API calls/minute limit")
print(f"This will take approximately {(end_date - start_date).days / 5} minutes")
print("="*60)

proceed = input("\nProceed with download? (y/n): ").lower()
if proceed != 'y':
    print("Aborted.")
    exit(0)

# Download data for each day using grouped daily endpoint
current_date = start_date
downloaded_dates = []
failed_dates = []
call_count = 0
start_time = time.time()

while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    output_file = output_dir / f"{date_str}.csv"
    
    # Skip weekends
    if current_date.weekday() >= 5:
        current_date += timedelta(days=1)
        continue
    
    # Skip if already downloaded
    if output_file.exists():
        print(f"⊙ {date_str}: Already downloaded, skipping")
        current_date += timedelta(days=1)
        continue
    
    try:
        print(f"Downloading {date_str}...", end=" ", flush=True)
        
        # Use grouped daily endpoint - gets all stocks for this date in one call
        # Endpoint: /v2/aggs/grouped/locale/us/market/stocks/{date}
        aggs = client.get_grouped_daily_aggs(date_str, locale="us", market_type="stocks")
        
        if aggs and len(aggs) > 0:
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'ticker': a.ticker,
                    'date': date_str,
                    'open': a.open,
                    'high': a.high,
                    'low': a.low,
                    'close': a.close,
                    'volume': a.volume,
                    'vwap': getattr(a, 'vwap', None),
                    'transactions': getattr(a, 'transactions', None),
                }
                for a in aggs
            ])
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            downloaded_dates.append(date_str)
            print(f"✓ ({len(df)} tickers)")
        else:
            print("✗ No data (holiday?)")
            failed_dates.append(date_str)
        
        call_count += 1
        
        # Rate limiting: 5 calls per minute on free tier
        if call_count % 5 == 0:
            elapsed = time.time() - start_time
            if elapsed < 60:
                wait_time = 60 - elapsed
                print(f"  ⏳ Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            start_time = time.time()
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        failed_dates.append(date_str)
        
        # If we hit rate limit, wait longer
        if "429" in str(e) or "rate limit" in str(e).lower():
            print("  ⏳ Rate limit hit, waiting 60s...")
            time.sleep(60)
            start_time = time.time()
    
    current_date += timedelta(days=1)

# Summary
print("\n" + "="*60)
print("Download Summary:")
print(f"  Successfully downloaded: {len(downloaded_dates)} days")
print(f"  Failed/Skipped: {len(failed_dates)} days")

# Combine all CSVs into one file
if downloaded_dates:
    combine = input("\nCombine all daily files into one CSV? (y/n): ").lower()
    
    if combine == 'y':
        print("\nCombining files...")
        all_dfs = []
        
        for date_str in downloaded_dates:
            file_path = output_dir / f"{date_str}.csv"
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        output_csv = output_dir.parent / "combined_daily_data.csv"
        combined_df.to_csv(output_csv, index=False)
        
        print(f"\n✓ Combined data saved to: {output_csv}")
        print(f"  Total rows: {len(combined_df):,}")
        print(f"  Unique tickers: {combined_df['ticker'].nunique()}")
        print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"\nFirst few rows:")
        print(combined_df.head(10))

print("\n✓ Done!")
