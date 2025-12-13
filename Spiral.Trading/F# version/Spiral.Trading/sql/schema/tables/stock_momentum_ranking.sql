CREATE TABLE IF NOT EXISTS stock_momentum_ranking (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    adj_close REAL,
    momentum_26w REAL,
    avg_dollar_volume_4w REAL,
    momentum_rank INTEGER,
    total_stocks INTEGER,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_smr_date ON stock_momentum_ranking(date);
CREATE INDEX IF NOT EXISTS idx_smr_momentum_rank ON stock_momentum_ranking(date, momentum_rank);
