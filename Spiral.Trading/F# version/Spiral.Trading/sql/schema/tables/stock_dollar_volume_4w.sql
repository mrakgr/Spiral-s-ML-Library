CREATE TABLE IF NOT EXISTS stock_dollar_volume_4w (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    total_dollar_volume REAL,
    trading_days INTEGER,
    avg_dollar_volume_4w REAL,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_sdv4w_date ON stock_dollar_volume_4w(date);
CREATE INDEX IF NOT EXISTS idx_sdv4w_avg_volume ON stock_dollar_volume_4w(avg_dollar_volume_4w);
