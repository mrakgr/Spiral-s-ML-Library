DROP VIEW IF EXISTS trading_calendar;
CREATE VIEW trading_calendar AS
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
        -- 26 weeks = 182 days, 25 weeks = 175 days
        WHERE date >= s1.date - INTERVAL '182 days'
        -- The -175 days condition ensures we have at least 25 weeks of historical data.
        -- This filters out the first ~6 months of the dataset where 26w lookback isn't possible.
        -- Safe assumption: US markets never close for more than 1 week consecutively.
        AND date < s1.date - INTERVAL '175 days'
    )
JOIN spy_dates s3
    ON s3.date = (
        SELECT MIN(date)
        FROM spy_dates
        -- 4 weeks = 28 days
        WHERE date >= s1.date - INTERVAL '28 days'
    );
