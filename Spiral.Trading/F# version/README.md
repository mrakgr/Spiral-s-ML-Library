# Spiral.Trading (F# Version)

A trading data analysis library for downloading and processing stock market data from Massive.

## Prerequisites

- .NET 9.0 SDK
- `api_key.json` file in the project root with your Massive API credentials:

```json
{
    "massive_api_key": "YOUR_API_KEY",
    "massive_s3_access_key": "YOUR_S3_ACCESS_KEY",
    "massive_s3_secret_key": "YOUR_S3_SECRET_KEY"
}
```

## Building

```bash
dotnet build
```

## CLI Commands

### Download Daily Aggregates

Downloads daily OHLCV data from Massive S3 storage as compressed CSV files.

```bash
dotnet run --project Spiral.Trading.Console -- download-bulk [options]
```

**Options:**
- `-s, --start-date <yyyy-MM-dd>` - Start date (default: 5 years ago)
- `-e, --end-date <yyyy-MM-dd>` - End date (default: today)
- `-p, --parallelism <int>` - Max parallel downloads (default: 30)

**Examples:**

```bash
# Download last 5 years of data
dotnet run --project Spiral.Trading.Console -- download-bulk

# Download specific date range
dotnet run --project Spiral.Trading.Console -- download-bulk -s 2024-01-01 -e 2024-12-11

# Download with lower parallelism
dotnet run --project Spiral.Trading.Console -- download-bulk -s 2024-12-01 -e 2024-12-11 -p 4
```

Output: `data/daily_aggregates/{yyyy-MM-dd}.csv.gz`

### Download Stock Splits

Downloads stock split information from the Massive API.

```bash
dotnet run --project Spiral.Trading.Console -- download-splits [options]
```

**Options:**
- `-s, --start-date <yyyy-MM-dd>` - Start date (default: 5 years ago)
- `-e, --end-date <yyyy-MM-dd>` - End date (default: none, includes all future splits)

**Examples:**

```bash
# Download all splits from the last 5 years
dotnet run --project Spiral.Trading.Console -- download-splits

# Download splits for a specific date range
dotnet run --project Spiral.Trading.Console -- download-splits -s 2024-01-01 -e 2024-12-11

# Download splits from a date onwards (no end date)
dotnet run --project Spiral.Trading.Console -- download-splits -s 2024-01-01
```

Output: `data/splits.json`

### Parse CSV Files

Parses downloaded CSV.gz files and displays summary information.

```bash
dotnet run --project Spiral.Trading.Console -- parse-csv [options]
```

**Options:**
- `-d, --directory <path>` - Directory containing .csv.gz files (default: data/daily_aggregates)
- `-f, --file <path>` - Single .csv.gz file to parse

**Examples:**

```bash
# Parse all files in default directory
dotnet run --project Spiral.Trading.Console -- parse-csv

# Parse a single file
dotnet run --project Spiral.Trading.Console -- parse-csv -f data/daily_aggregates/2024-12-09.csv.gz

# Parse files in custom directory
dotnet run --project Spiral.Trading.Console -- parse-csv -d /path/to/csv/files
```

## Project Structure

```
F# version/
├── Spiral.Trading/              # Core library
│   ├── Types.fs                 # Domain types
│   ├── Config.fs                # Configuration loading
│   ├── S3Download.fs            # S3 download functionality
│   ├── SplitDownload.fs         # Splits API client
│   └── CsvParsing.fs            # CSV parsing with FSharp.Data
├── Spiral.Trading.Console/      # CLI application
│   └── Program.fs
├── api_key.json                 # API credentials (not in git)
└── data/                        # Downloaded data
    ├── daily_aggregates/        # CSV files
    └── splits.json              # Splits data
```
