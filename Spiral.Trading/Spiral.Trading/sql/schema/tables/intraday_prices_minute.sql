-- Intraday minute-level price data
CREATE TABLE IF NOT EXISTS intraday_prices_minute (
    ticker VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    vwap DOUBLE,
    transactions INTEGER,
    PRIMARY KEY(ticker, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_minute_ticker ON intraday_prices_minute(ticker);
CREATE INDEX IF NOT EXISTS idx_minute_date ON intraday_prices_minute(CAST(timestamp AS DATE));
