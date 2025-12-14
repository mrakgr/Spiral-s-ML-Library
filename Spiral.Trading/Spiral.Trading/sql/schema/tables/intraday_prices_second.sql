-- Intraday second-level price data
CREATE TABLE IF NOT EXISTS intraday_prices_second (
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

CREATE INDEX IF NOT EXISTS idx_second_ticker ON intraday_prices_second(ticker);
CREATE INDEX IF NOT EXISTS idx_second_date ON intraday_prices_second(CAST(timestamp AS DATE));
