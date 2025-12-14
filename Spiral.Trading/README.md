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

Output: `data/splits.csv`

### Ingest Data

Ingests downloaded CSV files and splits into a SQLite database.

```bash
dotnet run --project Spiral.Trading.Console -- ingest-data [options]
```

**Options:**
- `-d, --database <path>` - SQLite database path (default: data/trading.db)
- `-c, --csv-dir <path>` - Directory containing .csv.gz files (default: data/daily_aggregates)
- `-s, --splits-file <path>` - JSON file containing splits (default: data/splits.json)

**Examples:**

```bash
# Ingest all data with defaults
dotnet run --project Spiral.Trading.Console -- ingest-data

# Ingest to a custom database
dotnet run --project Spiral.Trading.Console -- ingest-data -d /path/to/custom.db
```

**Features:**
- Tracks processed CSV files to avoid re-ingesting on subsequent runs
- Uses prepared statements and bulk load optimizations for fast ingestion
- Upserts splits (inserts new, updates existing) on each run
- Creates a `split_adjusted_prices` SQL view for efficient split-adjusted queries

### Plot Chart

Generates an interactive candlestick chart with volume for a given ticker.

```bash
dotnet run --project Spiral.Trading.Console -- plot-chart [options]
```

**Options:**
- `-t, --ticker <symbol>` - Stock ticker symbol (required)
- `-d, --database <path>` - SQLite database path (default: data/trading.db)
- `-o, --output <path>` - Output HTML file path (default: data/{ticker}_chart.html)
- `-w, --width <int>` - Chart width in pixels (default: 1200)
- `-h, --height <int>` - Chart height in pixels (default: 900)

**Examples:**

```bash
# Plot NVDA chart
dotnet run --project Spiral.Trading.Console -- plot-chart -t NVDA

# Plot with custom output path
dotnet run --project Spiral.Trading.Console -- plot-chart -t AAPL -o charts/apple.html

# Plot with custom dimensions
dotnet run --project Spiral.Trading.Console -- plot-chart -t MSFT -w 1600 -h 1000
```

**Features:**
- Split-adjusted prices calculated via SQL view
- Interactive candlestick chart with volume bars
- Output as standalone HTML file (uses Plotly.js)

### Plot DOM Chart

Generates a DOM (Direction of Momentum) indicator chart.

```bash
dotnet run --project Spiral.Trading.Console -- plot-dom [options]
```

**Options:**
- `-t, --ticker <symbol>` - Reference ticker to plot against (default: SPY)
- `-d, --database <path>` - DuckDB database path (default: data/trading.db)
- `-o, --output <path>` - Output HTML file path (default: data/dom_chart.html)
- `-w, --width <int>` - Chart width in pixels (default: 1200)
- `-h, --height <int>` - Chart height in pixels (default: 600)

**Examples:**

```bash
# Plot DOM chart with default SPY reference
dotnet run --project Spiral.Trading.Console -- plot-dom

# Plot DOM chart against QQQ
dotnet run --project Spiral.Trading.Console -- plot-dom -t QQQ

# Plot with custom output path
dotnet run --project Spiral.Trading.Console -- plot-dom -o charts/dom.html
```

**Features:**
- Market breadth indicator visualization
- Optional reference ticker overlay
- Output as standalone HTML file (uses Plotly.js)

## Project Structure

```
F# version/
├── Spiral.Trading/              # Core library
│   ├── Types.fs                 # Domain types
│   ├── Config.fs                # Configuration loading
│   ├── S3Download.fs            # S3 download functionality
│   ├── SplitDownload.fs         # Splits API client
│   ├── CsvParsing.fs            # CSV parsing with FSharp.Data
│   ├── Database.fs              # DuckDB database operations
│   ├── Plotting.fs              # Chart generation (candlestick, DOM)
│   └── sql/schema/              # SQL schema files
│       ├── tables/
│       │   ├── daily_prices.sql
│       │   ├── splits.sql
│       │   └── processed_files.sql
│       └── views/
│           ├── 01_split_adjusted_prices.sql
│           ├── 02_trading_calendar.sql
│           ├── 03_stock_momentum_26w.sql
│           ├── 04_stock_dollar_volume_4w.sql
│           ├── 05_stock_momentum_ranking.sql
│           ├── 06_stock_leaders.sql
│           ├── 07_stock_laggards.sql
│           └── 08_dom_indicator.sql
├── Spiral.Trading.Console/      # CLI application
│   └── Program.fs
├── api_key.json                 # API credentials (not in git)
└── data/                        # Downloaded data
    ├── daily_aggregates/        # CSV files
    ├── splits.csv              # Splits data
    └── trading.db               # SQLite database
```
