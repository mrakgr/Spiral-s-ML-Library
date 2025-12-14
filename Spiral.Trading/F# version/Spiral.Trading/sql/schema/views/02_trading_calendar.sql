-- Materialized table for trading calendar
-- Pre-computes date lookups for 26w and 4w ago
DROP VIEW IF EXISTS trading_calendar;
DROP TABLE IF EXISTS trading_calendar;
CREATE TABLE trading_calendar AS
WITH spy_dates AS (
    SELECT date
    FROM daily_prices
    WHERE ticker = 'SPY'
)
SELECT
    s1.date AS current_date,
    (SELECT MAX(date) FROM spy_dates WHERE date < s1.date) AS date_prev,
    s2.date AS date_26w_ago,
    s3.date AS date_4w_ago
FROM spy_dates s1
JOIN spy_dates s2
    ON s2.date = (
        SELECT MIN(date)
        FROM spy_dates
        WHERE date >= s1.date - INTERVAL '182 days'
        AND date < s1.date - INTERVAL '175 days'
    )
JOIN spy_dates s3
    ON s3.date = (
        SELECT MIN(date)
        FROM spy_dates
        WHERE date >= s1.date - INTERVAL '28 days'
    );
