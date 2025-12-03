import json
import boto3
import gzip
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os

# Load API credentials
with open("api_key.json") as f:
    api_keys = json.load(f)

# Get S3 credentials from Massive dashboard
# You need to get these from: https://polygon.io/dashboard/keys
# For now, we'll need to add them to api_key.json
s3_access_key = api_keys.get("massive_s3_access_key")
s3_secret_key = api_keys.get("massive_s3_secret_key")

if not s3_access_key or not s3_secret_key:
    print("ERROR: S3 credentials not found in api_key.json")
    print("Please add 'massive_s3_access_key' and 'massive_s3_secret_key' to api_key.json")
    print("Get them from: https://polygon.io/dashboard/keys")
    exit(1)

# Configure S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
    endpoint_url='https://files.massive.com'
)

bucket_name = 'flatfiles'

# Calculate date range for past 2 years
# Note: Many subscriptions have a 60-90 day delay for flat files
# Adjust end_date to account for this
end_date = datetime.now()
start_date = end_date - timedelta(weeks=52 * 5)  # 5 years before end_date

# Create output directory
output_dir = Path("data/daily_aggregates")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading daily aggregate data from {start_date.date()} to {end_date.date()}")
print(f"Output directory: {output_dir.absolute()}")

# Download files for each day
current_date = start_date
downloaded_files = []
failed_dates = []

while current_date <= end_date:
    # Format: us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
    year = current_date.year
    month = f"{current_date.month:02d}"
    date_str = current_date.strftime("%Y-%m-%d")
    
    s3_key = f"us_stocks_sip/day_aggs_v1/{year}/{month}/{date_str}.csv.gz"
    local_file = output_dir / f"{date_str}.csv.gz"
    
    try:
        print(f"Downloading {date_str}...", end=" ")
        s3.download_file(bucket_name, s3_key, str(local_file))
        downloaded_files.append(local_file)
        print("✓")
    except Exception as e:
        print(f"✗ (Error: {str(e)})")
        failed_dates.append(date_str)
    
    current_date += timedelta(days=1)

print(f"\n{'='*60}")
print(f"Download Summary:")
print(f"  Total files downloaded: {len(downloaded_files)}")
print(f"  Failed dates: {len(failed_dates)}")
if failed_dates:
    print(f"  Failed dates (likely weekends/holidays): {failed_dates[:10]}...")

# Optional: Decompress and combine into a single DataFrame (for smaller datasets)
combine_data = input("\nCombine all files into a single CSV? (y/n): ").lower() == 'y'

if combine_data:
    print("\nCombining all downloaded files...")
    all_data = []
    
    for file_path in downloaded_files:
        try:
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f)
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_csv = output_dir.parent / "combined_daily_data.csv"
        combined_df.to_csv(output_csv, index=False)
        print(f"Combined data saved to: {output_csv.absolute()}")
        print(f"Total rows: {len(combined_df):,}")
        print(f"Unique tickers: {combined_df['ticker'].nunique() if 'ticker' in combined_df.columns else 'N/A'}")
        print(f"\nFirst few rows:")
        print(combined_df.head())
    else:
        print("No data to combine.")

print("\n✓ Done!")
