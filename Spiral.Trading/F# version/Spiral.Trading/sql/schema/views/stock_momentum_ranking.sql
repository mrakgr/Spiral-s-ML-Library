DROP VIEW IF EXISTS stock_momentum_ranking;
CREATE VIEW stock_momentum_ranking AS
SELECT 
    m.ticker,
    m.date,
    m.adj_close,
    m.momentum_26w,
    v.avg_dollar_volume_4w,
    RANK() OVER (PARTITION BY m.date ORDER BY m.momentum_26w DESC) AS momentum_rank,
    COUNT(*) OVER (PARTITION BY m.date) AS total_stocks
FROM stock_momentum_26w m
JOIN stock_dollar_volume_4w v 
    ON v.ticker = m.ticker 
    AND v.date = m.date
WHERE v.avg_dollar_volume_4w >= 100000000;
