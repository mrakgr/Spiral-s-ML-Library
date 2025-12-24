-- Trades table for tick-level trade data
-- Timestamps are in UTC (nanosecond precision)
-- trade_date is the session date in Eastern time
CREATE SEQUENCE IF NOT EXISTS trades_id_seq;
CREATE TABLE IF NOT EXISTS trades (
    id BIGINT PRIMARY KEY DEFAULT nextval('trades_id_seq'),
    ticker VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    sip_timestamp TIMESTAMP_NS NOT NULL,
    participant_timestamp TIMESTAMP_NS NOT NULL,
    sequence_number BIGINT NOT NULL,
    price DOUBLE NOT NULL,
    size DOUBLE NOT NULL,
    exchange INTEGER NOT NULL,
    conditions INTEGER[],
    tape INTEGER
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker_trade_date ON trades(ticker, trade_date);
CREATE INDEX IF NOT EXISTS idx_trades_ticker_sip_timestamp ON trades(ticker, sip_timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_ticker_participant_timestamp ON trades(ticker, participant_timestamp);
