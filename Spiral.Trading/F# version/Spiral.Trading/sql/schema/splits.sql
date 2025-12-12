-- Stock split information
CREATE TABLE IF NOT EXISTS splits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    execution_date TEXT NOT NULL,  -- ISO 8601 format (YYYY-MM-DD)
    split_from REAL NOT NULL,
    split_to REAL NOT NULL,
    split_ratio REAL NOT NULL,
    UNIQUE(ticker, execution_date)
);

-- Index for efficient queries by ticker
CREATE INDEX IF NOT EXISTS idx_splits_ticker ON splits(ticker);

-- Index for efficient queries by execution date
CREATE INDEX IF NOT EXISTS idx_splits_execution_date ON splits(execution_date);
