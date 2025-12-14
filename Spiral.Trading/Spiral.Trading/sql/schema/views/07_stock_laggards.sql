-- View for bottom 10% momentum stocks (laggards)
DROP VIEW IF EXISTS stock_laggards;
CREATE VIEW stock_laggards AS
SELECT 
    ticker,
    date,
    momentum_rank,
    total_stocks
FROM stock_momentum_ranking
WHERE momentum_rank > total_stocks * 0.90;
