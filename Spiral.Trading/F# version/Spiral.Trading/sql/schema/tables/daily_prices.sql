-- Daily OHLCV price data
CREATE TABLE IF NOT EXISTS daily_prices (
    id INTEGER PRIMARY KEY,
    ticker VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(18, 4) NOT NULL,
    high DECIMAL(18, 4) NOT NULL,
    low DECIMAL(18, 4) NOT NULL,
    close DECIMAL(18, 4) NOT NULL,
    volume BIGINT NOT NULL,
    transactions BIGINT NOT NULL,
    UNIQUE(ticker, date)
);

-- Index for efficient queries by ticker
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices(ticker);

-- Index for efficient queries by date
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);

-- Index for efficient queries by ticker and date range
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date ON daily_prices(ticker, date);
