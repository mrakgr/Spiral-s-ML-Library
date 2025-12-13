CREATE VIEW IF NOT EXISTS trading_calendar AS
WITH spy_dates AS (
    SELECT date
    FROM daily_prices
    WHERE ticker = 'SPY'
)
SELECT 
    s1.date AS current_date,
    s2.date AS date_26w_ago,
    s3.date AS date_4w_ago
FROM spy_dates s1
JOIN spy_dates s2 
    ON s2.date = (
        SELECT MIN(date) 
        FROM spy_dates 
        WHERE date >= DATE(s1.date, '-26 weeks')
    )
JOIN spy_dates s3 
    ON s3.date = (
        SELECT MIN(date) 
        FROM spy_dates 
        WHERE date >= DATE(s1.date, '-4 weeks')
    );
