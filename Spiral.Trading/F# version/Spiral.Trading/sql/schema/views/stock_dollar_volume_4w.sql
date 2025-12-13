DROP VIEW IF EXISTS stock_dollar_volume_4w;
CREATE VIEW stock_dollar_volume_4w AS
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
JOIN trading_calendar tc ON p.date = tc.current_date;
