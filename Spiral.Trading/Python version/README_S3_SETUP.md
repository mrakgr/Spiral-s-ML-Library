# Massive S3 Flat Files Setup

## Getting S3 Credentials

1. Go to your Massive dashboard: https://polygon.io/dashboard/keys
2. Look for the "Flat Files" or "S3 Access" section
3. Copy your S3 Access Key and S3 Secret Key

## Update api_key.json

Add your S3 credentials to `api_key.json`:

```json
{
    "massive": "your_rest_api_key",
    "massive_s3_access_key": "your_s3_access_key",
    "massive_s3_secret_key": "your_s3_secret_key"
}
```

## Running the Bulk Download Script

```bash
python download_bulk_data.py
```

This script will:
- Download daily aggregate (OHLC) data for ALL US stocks
- For the past 2 years (730 days)
- Skip weekends/holidays automatically
- Save compressed files to `data/daily_aggregates/`
- Optionally combine all data into a single CSV

## File Structure

Downloaded files are organized as:
```
data/
└── daily_aggregates/
    ├── 2023-06-03.csv.gz
    ├── 2023-06-04.csv.gz
    ├── ...
    └── 2025-12-03.csv.gz
```

Each file contains OHLC data for ALL tickers for that specific date.

## Data Format

Each CSV file contains columns like:
- `ticker`: Stock symbol
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `vwap`: Volume-weighted average price
- `transactions`: Number of trades
- `timestamp`: Unix timestamp

## Notes

- Files are in gzip-compressed CSV format
- Each daily file contains data for ALL stocks (thousands of tickers)
- Total download size for 2 years: ~10-20 GB compressed
- Market is closed on weekends/holidays - those dates will fail gracefully
