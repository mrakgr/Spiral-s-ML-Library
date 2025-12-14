-- Materialized table for 4-week average dollar volume
-- Only stores ticker, date, and avg_dollar_volume_4w
DROP VIEW IF EXISTS stock_dollar_volume_4w;
DROP TABLE IF EXISTS stock_dollar_volume_4w;
CREATE TABLE stock_dollar_volume_4w AS
WITH volume_data AS (
    SELECT
        p.ticker,
        tc.current_date AS date,
        (
            SELECT SUM(p2.adj_close * p2.adj_volume)
            FROM split_adjusted_prices p2
            WHERE p2.ticker = p.ticker
            AND p2.date >= tc.date_4w_ago
            AND p2.date <= tc.current_date
        ) AS total_dollar_volume,
        (
            SELECT COUNT(*)
            FROM split_adjusted_prices p2
            WHERE p2.ticker = p.ticker
            AND p2.date >= tc.date_4w_ago
            AND p2.date <= tc.current_date
        ) AS trading_days
    FROM split_adjusted_prices p
    JOIN trading_calendar tc ON p.date = tc.current_date
)
SELECT
    ticker,
    date,
    total_dollar_volume / NULLIF(trading_days, 0) AS avg_dollar_volume_4w
FROM volume_data;
