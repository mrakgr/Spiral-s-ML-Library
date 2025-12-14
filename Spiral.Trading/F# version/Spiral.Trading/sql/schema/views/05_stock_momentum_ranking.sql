-- Materialized table for momentum ranking
-- Only stores ticker, date, momentum_rank, and total_stocks
-- (momentum_26w available in stock_momentum_26w, avg_dollar_volume_4w in stock_dollar_volume_4w)
DROP TABLE IF EXISTS stock_momentum_ranking;
CREATE TABLE stock_momentum_ranking AS
SELECT 
    m.ticker,
    m.date,
    RANK() OVER (PARTITION BY m.date ORDER BY m.momentum_26w DESC) AS momentum_rank,
    COUNT(*) OVER (PARTITION BY m.date) AS total_stocks
FROM stock_momentum_26w m
JOIN stock_dollar_volume_4w v 
    ON v.ticker = m.ticker 
    AND v.date = m.date
WHERE v.avg_dollar_volume_4w >= 100000000;
