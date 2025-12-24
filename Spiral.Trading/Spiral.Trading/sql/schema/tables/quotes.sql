-- Quotes table for NBBO quote data
-- Timestamps are in UTC (nanosecond precision)
-- trade_date is the session date in Eastern time
CREATE TABLE IF NOT EXISTS quotes (
    ticker VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    sip_timestamp TIMESTAMP_NS NOT NULL,
    participant_timestamp TIMESTAMP_NS NOT NULL,
    sequence_number BIGINT NOT NULL,
    bid_price DOUBLE NOT NULL,
    bid_size DOUBLE NOT NULL,
    bid_exchange INTEGER NOT NULL,
    ask_price DOUBLE NOT NULL,
    ask_size DOUBLE NOT NULL,
    ask_exchange INTEGER NOT NULL,
    conditions INTEGER[],
    indicators INTEGER[],
    tape INTEGER
);

CREATE INDEX IF NOT EXISTS idx_quotes_ticker ON quotes(ticker);
CREATE INDEX IF NOT EXISTS idx_quotes_trade_date ON quotes(trade_date);
