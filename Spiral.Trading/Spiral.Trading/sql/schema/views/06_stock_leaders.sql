-- View for top 10% momentum stocks (leaders)
DROP VIEW IF EXISTS stock_leaders;
CREATE VIEW stock_leaders AS
SELECT 
    ticker,
    date,
    momentum_rank,
    total_stocks
FROM stock_momentum_ranking
WHERE momentum_rank <= total_stocks * 0.10;
