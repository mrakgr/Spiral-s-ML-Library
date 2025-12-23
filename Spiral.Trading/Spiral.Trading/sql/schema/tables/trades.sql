-- Trades table for tick-level trade data
CREATE TABLE IF NOT EXISTS trades (
    ticker VARCHAR NOT NULL,
    sip_timestamp BIGINT NOT NULL,
    participant_timestamp BIGINT NOT NULL,
    sequence_number BIGINT NOT NULL,
    price DOUBLE NOT NULL,
    size DOUBLE NOT NULL,
    exchange INTEGER NOT NULL,
    conditions INTEGER[],
    tape INTEGER,
    PRIMARY KEY (ticker, sip_timestamp, sequence_number)
);
