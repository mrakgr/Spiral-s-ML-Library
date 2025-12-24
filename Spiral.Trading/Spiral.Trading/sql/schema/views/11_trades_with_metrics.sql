-- Macro for computing trade metrics with configurable time window
-- window_seconds: number of seconds for rolling window, or NULL for full session
-- Returns trades with VWAP, VWSTD, and volume breakdown by side
-- Partitions by session_date to isolate sessions

DROP MACRO TABLE IF EXISTS trades_with_metrics;
CREATE MACRO trades_with_metrics(window_seconds) AS TABLE
SELECT 
    ticker,
    session_date,
    sip_timestamp,
    participant_timestamp,
    sequence_number,
    price,
    size,
    exchange,
    conditions,
    tape,
    bid_price,
    ask_price,
    bid_size,
    ask_size,
    side,
    
    -- Running VWAP
    SUM(price * size) OVER w / SUM(size) OVER w AS vwap,
    
    -- Running VWSTD: sqrt(VWAP(price²) - VWAP(price)²)
    SQRT(GREATEST(0,
        SUM(price * price * size) OVER w / SUM(size) OVER w 
        - POWER(SUM(price * size) OVER w / SUM(size) OVER w, 2)
    )) AS vwstd,
    
    -- Volume by side
    SUM(CASE WHEN side = 'BUY' THEN size ELSE 0 END) OVER w AS ask_volume,
    SUM(CASE WHEN side = 'SELL' THEN size ELSE 0 END) OVER w AS bid_volume,
    SUM(CASE WHEN side = 'MID' THEN size ELSE 0 END) OVER w AS mid_volume,
    SUM(size) OVER w AS total_volume

FROM trades_with_quotes
WINDOW w AS (
    PARTITION BY ticker, session_date
    ORDER BY participant_timestamp
    RANGE BETWEEN 
        COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND PRECEDING 
        AND CURRENT ROW
);
