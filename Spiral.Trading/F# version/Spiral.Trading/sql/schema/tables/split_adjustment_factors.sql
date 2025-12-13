CREATE TABLE IF NOT EXISTS split_adjustment_factors (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    adj_factor REAL NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_saf_date ON split_adjustment_factors(date);
