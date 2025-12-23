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
- `-p, --parallelism <int>` - Max parallel downloads (default: 10)

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

### Download Intraday Data

Downloads intraday (minute or second) aggregate bars for specific tickers via the Polygon REST API. Can download for individual tickers or automatically fetch data for all stocks in play.

```bash
dotnet run --project Spiral.Trading.Console -- download-intraday [options]
```

**Options:**
- `-t, --ticker <symbol>` - Stock ticker symbol (use with date range)
- `-s, --start-date <yyyy-MM-dd>` - Start date (default: 1 week ago)
- `-e, --end-date <yyyy-MM-dd>` - End date. If omitted with -t, only start date is downloaded
- `-d, --database <path>` - DuckDB database path for SIP lookup (default: data/trading.db)
- `-o, --output-dir <path>` - Output directory (default: data/intraday)
- `-p, --parallelism <int>` - Max parallel downloads (default: 5)
- `--timespan <string>` - Aggregate timespan: 'minute' or 'second' (default: minute)
- `--from-sip` - Download for all stocks in play from database
- `-r, --min-rvol <float>` - Min RVOL filter for SIP (default: 3)
- `-g, --min-gap-pct <float>` - Min gap % for SIP (default: 0.05)
- `-v, --min-dollar-volume <float>` - Min avg dollar volume in millions for SIP (default: 100)

**Examples:**

```bash
# Download minute data for a specific ticker
dotnet run --project Spiral.Trading.Console -- download-intraday -t NVDA -s 2024-12-01 -e 2024-12-11

# Download second aggregates for a ticker
dotnet run --project Spiral.Trading.Console -- download-intraday -t AAPL -s 2024-12-10 --timespan second

# Download minute data for all stocks in play (from database)
dotnet run --project Spiral.Trading.Console -- download-intraday --from-sip -s 2024-12-01 -e 2024-12-11

# Download for SIPs with custom filters
dotnet run --project Spiral.Trading.Console -- download-intraday --from-sip -r 5 -g 0.10 -s 2024-12-01
```

Output: `data/intraday/{timespan}/{ticker}/{date}.json`

### Download Trades Data

Downloads tick-level trades data for a specific ticker via the Polygon REST API. Each trade record includes price, size, exchange, conditions, and precise timestamps.

```bash
dotnet run --project Spiral.Trading.Console -- download-trades [options]
```

**Options:**
- `-t, --ticker <symbol>` - Stock ticker symbol (required)
- `-s, --start-date <yyyy-MM-dd>` - Start date (required)
- `-e, --end-date <yyyy-MM-dd>` - End date. If omitted, only start date is downloaded
- `-o, --output-dir <path>` - Output directory (default: data/trades)
- `-p, --parallelism <int>` - Max parallel downloads (default: 5)

**Examples:**

```bash
# Download trades for a single day
dotnet run --project Spiral.Trading.Console -- download-trades -t NVDA -s 2024-12-20

# Download trades for a date range
dotnet run --project Spiral.Trading.Console -- download-trades -t NVDA -s 2024-12-15 -e 2024-12-20

# Download with custom output directory
dotnet run --project Spiral.Trading.Console -- download-trades -t AAPL -s 2024-12-20 -o data/my_trades
```

Output: `data/trades/{ticker}/{date}.json`

### Download Quotes Data

Downloads NBBO (National Best Bid and Offer) quotes data for a specific ticker via the Polygon REST API. Each quote record includes bid/ask prices, sizes, exchanges, and timestamps.

```bash
dotnet run --project Spiral.Trading.Console -- download-quotes [options]
```

**Options:**
- `-t, --ticker <symbol>` - Stock ticker symbol (required)
- `-s, --start-date <yyyy-MM-dd>` - Start date (required)
- `-e, --end-date <yyyy-MM-dd>` - End date. If omitted, only start date is downloaded
- `-o, --output-dir <path>` - Output directory (default: data/quotes)
- `-p, --parallelism <int>` - Max parallel downloads (default: 5)
- `--pretty` - Output JSON with indentation (pretty print)

**Examples:**

```bash
# Download quotes for a single day
dotnet run --project Spiral.Trading.Console -- download-quotes -t NVDA -s 2024-12-20

# Download quotes for a date range
dotnet run --project Spiral.Trading.Console -- download-quotes -t NVDA -s 2024-12-15 -e 2024-12-20

# Download with pretty-printed JSON
dotnet run --project Spiral.Trading.Console -- download-quotes -t AAPL -s 2024-12-20 --pretty
```

Output: `data/quotes/{ticker}/{date}.json`

### Ingest Data

Ingests downloaded CSV files and splits into a DuckDB database.

```bash
dotnet run --project Spiral.Trading.Console -- ingest-data [options]
```

**Options:**
- `-d, --database <path>` - DuckDB database path (default: data/trading.db)
- `-c, --csv-dir <path>` - Directory containing .csv.gz files (default: data/daily_aggregates)
- `-s, --splits-file <path>` - CSV file containing splits (default: data/splits.csv)

**Examples:**

```bash
# Ingest all data with defaults
dotnet run --project Spiral.Trading.Console -- ingest-data

# Ingest to a custom database
dotnet run --project Spiral.Trading.Console -- ingest-data -d /path/to/custom.db
```

**Features:**
- Uses DuckDB's native CSV reader for fast bulk ingestion
- Upserts splits (inserts new, updates existing) on each run
- Materializes derived tables (split-adjusted prices, momentum rankings, etc.)

### Ingest Intraday Data

Ingests downloaded intraday JSON files into the DuckDB database.

```bash
dotnet run --project Spiral.Trading.Console -- ingest-intraday [options]
```

**Options:**
- `-d, --database <path>` - DuckDB database path (default: data/trading.db)
- `-i, --input-dir <path>` - Input directory for intraday data (default: data/intraday)
- `--timespan <string>` - Filter by timespan: 'minute', 'second', or 'all' (default: all)

**Examples:**

```bash
# Ingest all intraday data (minute and second)
dotnet run --project Spiral.Trading.Console -- ingest-intraday

# Ingest only minute data
dotnet run --project Spiral.Trading.Console -- ingest-intraday --timespan minute

# Ingest only second data
dotnet run --project Spiral.Trading.Console -- ingest-intraday --timespan second

# Ingest from a custom directory
dotnet run --project Spiral.Trading.Console -- ingest-intraday -i /path/to/intraday
```

**Features:**
- Uses DuckDB's native JSON reader with glob patterns for fast bulk ingestion
- Separate tables for minute (`intraday_prices_minute`) and second (`intraday_prices_second`) data
- Upserts on conflict (updates existing bars)

### Refresh Views

Refreshes only the SQL views without rematerializing the derived tables. Use this when you've modified view definitions but don't need to recompute the underlying materialized tables.

```bash
dotnet run --project Spiral.Trading.Console -- refresh-views [options]
```

**Options:**
- `-d, --database <path>` - DuckDB database path (default: data/trading.db)

**Examples:**

```bash
# Refresh views with default database
dotnet run --project Spiral.Trading.Console -- refresh-views

# Refresh views for a custom database
dotnet run --project Spiral.Trading.Console -- refresh-views -d /path/to/custom.db
```

### Plot Chart

Generates an interactive candlestick chart with volume for a given ticker.

```bash
dotnet run --project Spiral.Trading.Console -- plot-chart [options]
```

**Options:**
- `-t, --ticker <symbol>` - Stock ticker symbol (required)
- `-d, --database <path>` - DuckDB database path (default: data/trading.db)
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

### Plot Intraday Chart

Generates an interactive intraday candlestick chart with volume and VWAP indicators.

```bash
dotnet run --project Spiral.Trading.Console -- plot-intraday [options]
```

**Options:**
- `-t, --ticker <symbol>` - Stock ticker symbol (required)
- `-s, --date <yyyy-MM-dd>` - Date to plot (required)
- `-d, --database <path>` - DuckDB database path (default: data/trading.db)
- `-o, --output <path>` - Output HTML file path (default: data/{ticker}_{date}_intraday.html)
- `-w, --width <int>` - Chart width in pixels (default: 1200)
- `-h, --height <int>` - Chart height in pixels (default: 900)
- `--timespan <string>` - Aggregate timespan: 'minute' or 'second' (default: minute)

**Examples:**

```bash
# Plot NVDA intraday chart for a specific date
dotnet run --project Spiral.Trading.Console -- plot-intraday -t NVDA -s 2024-12-12

# Plot second-level data
dotnet run --project Spiral.Trading.Console -- plot-intraday -t AAPL -s 2024-12-12 --timespan second

# Plot with custom output path
dotnet run --project Spiral.Trading.Console -- plot-intraday -t MSFT -s 2024-12-12 -o charts/msft_intraday.html
```

**Features:**
- Candlestick chart with OHLC prices
- Volume bars at the bottom
- Session VWAP (orange line) - cumulative volume-weighted average price, commonly used as institutional execution benchmark
- Bar VWAP (purple dots) - individual bar VWAP values
- Unified hover mode for easy data inspection
- Output as standalone HTML file (uses Plotly.js)

### Stocks In Play

Lists top stocks in play for a date range based on relative volume, opening gap, and liquidity.

```bash
dotnet run --project Spiral.Trading.Console -- stocks-in-play [options]
```

**Options:**
- `-s, --start-date <yyyy-MM-dd>` - Start date (default: 1 week ago)
- `-e, --end-date <yyyy-MM-dd>` - End date (default: today)
- `-d, --database <path>` - DuckDB database path (default: data/trading.db)
- `-r, --min-rvol <float>` - Minimum relative volume (default: 3)
- `-g, --min-gap-pct <float>` - Minimum gap percentage as decimal (default: 0.05 for 5%)
- `-v, --min-dollar-volume <float>` - Minimum avg dollar volume in millions (default: 100)

**Examples:**

```bash
# List stocks in play for the past week (default filters)
dotnet run --project Spiral.Trading.Console -- stocks-in-play

# List stocks in play for a specific date range
dotnet run --project Spiral.Trading.Console -- stocks-in-play -s 2024-12-01 -e 2024-12-11

# Find stocks with higher volatility (5x RVOL, 10% gap)
dotnet run --project Spiral.Trading.Console -- stocks-in-play -r 5 -g 0.10

# Include smaller-cap stocks ($50M+ avg volume instead of $100M)
dotnet run --project Spiral.Trading.Console -- stocks-in-play -v 50

# Combine all filters: aggressive settings for small caps
dotnet run --project Spiral.Trading.Console -- stocks-in-play -r 2 -g 0.03 -v 25
```

**Default Criteria:**
- Liquidity: $100M+ average daily dollar volume (4-week)
- Relative Volume (RVOL): >= 3x normal volume
- Opening Gap: >= 5% from previous close
- Ranked by composite score (RVOL + gap magnitude)
- Top 10 stocks per day

### List Conditions

Lists trade and quote condition codes from the Polygon API. These codes appear in the `conditions` field of trades data and indicate special circumstances like extended hours trades, odd lots, or intermarket sweeps.

```bash
dotnet run --project Spiral.Trading.Console -- list-conditions [options]
```

**Options:**
- `-a, --asset-class <string>` - Asset class filter: stocks, options, crypto, fx (default: stocks)
- `-d, --data-type <string>` - Data type filter: trade, quote (default: trade)

**Examples:**

```bash
# List all stock trade conditions (default)
dotnet run --project Spiral.Trading.Console -- list-conditions

# List quote conditions for stocks
dotnet run --project Spiral.Trading.Console -- list-conditions -d quote

# List trade conditions for options
dotnet run --project Spiral.Trading.Console -- list-conditions -a options
```

**Output columns:**
- **ID** - Condition code number (appears in trades data `conditions` field)
- **Name** - Human-readable description
- **Type** - Category (sale_condition, trade_thru_exempt, etc.)
- **CTA/UTP** - Exchange-specific codes
- **Hi/Lo, Op/Cl, Volume** - Whether trades with this condition update high/low, open/close, and volume calculations

**Common conditions:**
- **37 (Odd Lot Trade)** - Trades < 100 shares, excluded from hi/lo and open/close
- **12 (Form T/Extended Hours)** - Pre-market and after-hours trades
- **14 (Intermarket Sweep)** - Large orders swept across multiple exchanges
- **41 (Trade Thru Exempt)** - Exempt from trade-through rules

## Project Structure

```
Spiral.Trading/
├── Spiral.Trading/              # Core library
│   ├── Types.fs                 # Domain types
│   ├── Config.fs                # Configuration loading
│   ├── S3Download.fs            # S3 bulk download (daily aggregates)
│   ├── SplitDownload.fs         # Splits API client
│   ├── IntradayDownload.fs      # Intraday API client (minute/second bars)
│   ├── TradesDownload.fs        # Trades API client (tick-level data)
│   ├── QuotesDownload.fs        # Quotes API client (NBBO data)
│   ├── Conditions.fs            # Trade condition codes API client
│   ├── Database.fs              # DuckDB database operations
│   ├── Plotting.fs              # Chart generation (candlestick, DOM)
│   └── sql/schema/              # SQL schema files
│       ├── tables/              # Base tables
│       │   ├── daily_prices.sql
│       │   └── splits.sql
│       ├── materialized/        # Materialized tables (slow to rebuild)
│       │   ├── 01_split_adjusted_prices.sql
│       │   ├── 02_trading_calendar.sql
│       │   ├── 03_stock_momentum_26w.sql
│       │   ├── 04_stock_dollar_volume_4w.sql
│       │   └── 05_stock_momentum_ranking.sql
│       └── views/               # Views/macros (fast to refresh)
│           ├── 06_stock_leaders.sql
│           ├── 07_stock_laggards.sql
│           ├── 08_dom_indicator.sql
│           └── 09_stocks_in_play.sql
├── Spiral.Trading.Console/      # CLI application
│   └── Program.fs
├── api_key.json                 # API credentials (not in git)
└── data/                        # Downloaded data
    ├── daily_aggregates/        # Daily OHLCV CSV files
    ├── intraday/                # Intraday data (minute/second JSON)
    ├── trades/                  # Tick-level trades data (JSON)
    ├── quotes/                  # NBBO quotes data (JSON)
    ├── splits.csv               # Splits data
    └── trading.db               # DuckDB database
```
