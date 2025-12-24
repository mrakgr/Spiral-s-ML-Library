-- Quotes table for NBBO quote data
-- Timestamps are in UTC (nanosecond precision)
-- session_date is the trading session date in Eastern time
CREATE TABLE IF NOT EXISTS quotes (
    ticker VARCHAR NOT NULL,
    session_date DATE NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_quotes_session_date ON quotes(session_date);
