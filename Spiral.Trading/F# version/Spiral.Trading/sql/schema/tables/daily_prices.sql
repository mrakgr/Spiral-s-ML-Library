-- Daily OHLCV price data
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume BIGINT NOT NULL,
    transactions BIGINT NOT NULL,
    PRIMARY KEY(ticker, date)
);

-- Index for efficient queries by ticker
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices(ticker);

-- Index for efficient queries by date
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);

-- Index for efficient queries by ticker and date range
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date ON daily_prices(ticker, date);
