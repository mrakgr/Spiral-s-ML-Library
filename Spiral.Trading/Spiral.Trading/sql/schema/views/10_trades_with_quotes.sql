-- View for trades with matched quotes and computed side
-- Uses ASOF JOIN on quote.sip_timestamp <= trade.participant_timestamp
-- Joins on session_date to ensure we only match within the same session
CREATE TYPE IF NOT EXISTS trade_side AS ENUM ('BUY', 'SELL', 'MID');

DROP VIEW IF EXISTS trades_with_quotes;
CREATE VIEW trades_with_quotes AS
SELECT 
    t.id,
    t.ticker,
    t.session_date,
    t.sip_timestamp,
    t.participant_timestamp,
    t.sequence_number,
    t.price,
    t.size,
    t.exchange,
    t.conditions,
    t.tape,
    q.bid_price,
    q.ask_price,
    q.bid_size,
    q.ask_size,
    CASE 
        WHEN t.price >= q.ask_price THEN 'BUY'::trade_side
        WHEN t.price <= q.bid_price THEN 'SELL'::trade_side
        ELSE 'MID'::trade_side
    END AS side
FROM trades t
ASOF JOIN quotes q 
    ON t.ticker = q.ticker 
    AND t.session_date = q.session_date
    AND t.participant_timestamp >= q.sip_timestamp;
