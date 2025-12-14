-- Stock split information
CREATE TABLE IF NOT EXISTS splits (
    ticker VARCHAR NOT NULL,
    execution_date DATE NOT NULL,
    split_from DOUBLE NOT NULL,
    split_to DOUBLE NOT NULL,
    split_ratio DOUBLE NOT NULL,
    PRIMARY KEY(ticker, execution_date)
);

-- Index for efficient queries by ticker
CREATE INDEX IF NOT EXISTS idx_splits_ticker ON splits(ticker);

-- Index for efficient queries by execution date
CREATE INDEX IF NOT EXISTS idx_splits_execution_date ON splits(execution_date);
