-- Daily OHLCV price data
CREATE TABLE IF NOT EXISTS daily_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,  -- ISO 8601 format (YYYY-MM-DD)
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    transactions INTEGER NOT NULL,
    UNIQUE(ticker, date)
);

-- Index for efficient queries by ticker
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices(ticker);

-- Index for efficient queries by date
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);

-- Index for efficient queries by ticker and date range
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date ON daily_prices(ticker, date);
