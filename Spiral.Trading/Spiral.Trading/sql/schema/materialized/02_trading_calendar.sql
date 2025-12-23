-- Materialized table for trading calendar
-- Pre-computes date lookups for previous day, 26w and 4w ago using ASOF JOIN
DROP TABLE IF EXISTS trading_calendar;
CREATE TABLE trading_calendar AS
WITH spy_dates AS (
    SELECT date FROM daily_prices WHERE ticker = 'SPY'
)
SELECT
    s1.date AS current_date,
    s_prev.date AS date_prev,
    s_26w.date AS date_26w_ago,
    s_4w.date AS date_4w_ago
FROM spy_dates s1
ASOF JOIN spy_dates s_prev 
    ON s_prev.date < s1.date 
ASOF JOIN spy_dates s_26w 
    ON s_26w.date <= s1.date - INTERVAL '182 days'
ASOF JOIN spy_dates s_4w 
    ON s_4w.date <= s1.date - INTERVAL '28 days';
