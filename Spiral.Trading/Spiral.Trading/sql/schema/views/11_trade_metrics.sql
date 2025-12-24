-- Macro for computing trade metrics with configurable time window
-- window_seconds: number of seconds for rolling window, or NULL for full session
-- Returns trade id with VWAP, VWSTD, and volume breakdown by side
-- Partitions by session_date to isolate sessions
-- Orders by sip_timestamp (when trade is visible) for realistic simulation
-- Filters by participant_timestamp to only include trades that occurred within window
-- Join with trades_with_quotes on id to get full trade info

DROP MACRO TABLE IF EXISTS trade_metrics;
CREATE MACRO trade_metrics(window_seconds) AS TABLE
WITH windowed AS (
    SELECT 
        *,
        LAST_VALUE(sip_timestamp) OVER w AS current_sip_ts
    FROM trades_with_quotes
    WINDOW w AS (
        PARTITION BY ticker, session_date
        ORDER BY sip_timestamp
        RANGE BETWEEN 
            COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND PRECEDING 
            AND CURRENT ROW
    )
)
SELECT 
    id,
    
    -- Running VWAP (only trades with participant_timestamp in window)
    SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
        THEN price * size ELSE 0 END) OVER w 
    / NULLIF(SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
        THEN size ELSE 0 END) OVER w, 0) AS vwap,
    
    -- Running VWSTD: sqrt(VWAP(price²) - VWAP(price)²)
    SQRT(GREATEST(0,
        SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
            THEN price * price * size ELSE 0 END) OVER w 
        / NULLIF(SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
            THEN size ELSE 0 END) OVER w, 0)
        - POWER(
            SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
                THEN price * size ELSE 0 END) OVER w 
            / NULLIF(SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
                THEN size ELSE 0 END) OVER w, 0)
        , 2)
    )) AS vwstd,
    
    -- Volume by side (filtered)
    SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
        AND side = 'BUY' THEN size ELSE 0 END) OVER w AS ask_volume,
    SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
        AND side = 'SELL' THEN size ELSE 0 END) OVER w AS bid_volume,
    SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
        AND side = 'MID' THEN size ELSE 0 END) OVER w AS mid_volume,
    SUM(CASE WHEN participant_timestamp >= current_sip_ts - COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND 
        THEN size ELSE 0 END) OVER w AS total_volume

FROM windowed
WINDOW w AS (
    PARTITION BY ticker, session_date
    ORDER BY sip_timestamp
    RANGE BETWEEN 
        COALESCE(window_seconds, 86400) * INTERVAL 1 SECOND PRECEDING 
        AND CURRENT ROW
);
