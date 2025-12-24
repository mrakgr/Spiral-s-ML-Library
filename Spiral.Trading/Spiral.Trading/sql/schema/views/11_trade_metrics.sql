-- Macro for computing trade metrics with configurable time window
-- window_seconds: number of seconds for rolling window, or NULL for full session
-- Returns trade id with VWAP, VWSTD, and volume breakdown by side
-- Partitions by session_date to isolate sessions
-- Orders by sip_timestamp (when trade is visible) for realistic simulation
-- Join with trades_with_quotes on id to get full trade info

DROP MACRO TABLE IF EXISTS trade_metrics;
CREATE MACRO trade_metrics(window_seconds) AS TABLE
SELECT 
    id,
    
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
    ORDER BY sip_timestamp
    RANGE BETWEEN 
        COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND PRECEDING 
        AND CURRENT ROW
);
