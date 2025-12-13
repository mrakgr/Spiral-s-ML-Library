DROP VIEW IF EXISTS stock_laggards;
CREATE VIEW stock_laggards AS
SELECT 
    ticker,
    date,
    adj_close,
    momentum_26w,
    avg_dollar_volume_4w,
    momentum_rank,
    total_stocks
FROM stock_momentum_ranking
WHERE momentum_rank > total_stocks * 0.90;
