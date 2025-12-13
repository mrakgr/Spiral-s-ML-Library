DROP VIEW IF EXISTS stock_dollar_volume_4w;
CREATE VIEW stock_dollar_volume_4w AS
SELECT 
    p.ticker,
    tc.current_date AS date,
    SUM(p.adj_close * p.adj_volume) AS total_dollar_volume,
    COUNT(*) AS trading_days,
    SUM(p.adj_close * p.adj_volume) / COUNT(*) AS avg_dollar_volume_4w
FROM split_adjusted_prices p
JOIN trading_calendar tc 
    ON p.date >= tc.date_4w_ago 
    AND p.date <= tc.current_date
GROUP BY p.ticker, tc.current_date;
