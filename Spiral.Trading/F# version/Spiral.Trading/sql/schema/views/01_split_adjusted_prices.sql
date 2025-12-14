-- Materialized table for split-adjusted prices
-- Only stores adjusted values, not original OHLCV (available in daily_prices)
DROP TABLE IF EXISTS split_adjusted_prices;
CREATE TABLE split_adjusted_prices AS
WITH split_factors AS (
    SELECT 
        dp.ticker,
        dp.date,
        dp.open,
        dp.high,
        dp.low,
        dp.close,
        dp.volume,
        COALESCE(
            (SELECT EXP(SUM(LN(s.split_ratio)))
             FROM splits s
             WHERE s.ticker = dp.ticker
             AND s.execution_date > dp.date),
            1.0
        ) AS adj_factor
    FROM daily_prices dp
)
SELECT
    ticker,
    date,
    open / adj_factor AS adj_open,
    high / adj_factor AS adj_high,
    low / adj_factor AS adj_low,
    close / adj_factor AS adj_close,
    CAST(volume * adj_factor AS BIGINT) AS adj_volume
FROM split_factors;
