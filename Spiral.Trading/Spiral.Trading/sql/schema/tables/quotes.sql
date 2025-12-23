-- Quotes table for NBBO quote data
CREATE TABLE IF NOT EXISTS quotes (
    ticker VARCHAR NOT NULL,
    sip_timestamp BIGINT NOT NULL,
    participant_timestamp BIGINT NOT NULL,
    sequence_number BIGINT NOT NULL,
    bid_price DOUBLE NOT NULL,
    bid_size DOUBLE NOT NULL,
    bid_exchange INTEGER NOT NULL,
    ask_price DOUBLE NOT NULL,
    ask_size DOUBLE NOT NULL,
    ask_exchange INTEGER NOT NULL,
    conditions INTEGER[],
    indicators INTEGER[],
    tape INTEGER,
    PRIMARY KEY (ticker, sip_timestamp, sequence_number)
);
