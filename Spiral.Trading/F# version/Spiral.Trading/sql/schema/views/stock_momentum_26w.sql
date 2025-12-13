DROP VIEW IF EXISTS stock_momentum_26w;
CREATE VIEW stock_momentum_26w AS
SELECT 
    p.ticker,
    tc.current_date AS date,
    p.adj_close,
    p_26w.adj_close AS adj_close_26w_ago,
    (p.adj_close - p_26w.adj_close) / p_26w.adj_close AS momentum_26w
FROM split_adjusted_prices p
JOIN trading_calendar tc ON p.date = tc.current_date
JOIN split_adjusted_prices p_26w 
    ON p_26w.ticker = p.ticker 
    AND p_26w.date = tc.date_26w_ago;
