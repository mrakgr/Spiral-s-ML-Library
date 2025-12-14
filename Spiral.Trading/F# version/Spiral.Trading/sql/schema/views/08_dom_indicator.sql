-- DOM (Direction of Momentum) indicator view
-- Uses LAG to get previous day's rank without joining stock_momentum_ranking twice
DROP VIEW IF EXISTS dom_indicator;
CREATE VIEW dom_indicator AS
WITH ranked_with_prev AS (
    SELECT 
        r.ticker,
        r.date,
        r.momentum_rank,
        r.total_stocks,
        LAG(r.momentum_rank) OVER (PARTITION BY r.ticker ORDER BY r.date) AS prev_momentum_rank,
        LAG(r.total_stocks) OVER (PARTITION BY r.ticker ORDER BY r.date) AS prev_total_stocks
    FROM stock_momentum_ranking r
),
daily_returns AS (
    SELECT 
        rp.ticker,
        rp.date,
        GREATEST(-0.50, LEAST(1.00, 
            (p.adj_close - p_prev.adj_close) / p_prev.adj_close
        )) AS daily_return,
        rp.prev_momentum_rank,
        rp.prev_total_stocks
    FROM ranked_with_prev rp
    JOIN trading_calendar tc ON rp.date = tc.current_date
    JOIN split_adjusted_prices p 
        ON p.ticker = rp.ticker 
        AND p.date = rp.date
    JOIN split_adjusted_prices p_prev 
        ON p_prev.ticker = rp.ticker 
        AND p_prev.date = tc.date_prev
    WHERE rp.prev_momentum_rank IS NOT NULL
),
leader_laggard_returns AS (
    SELECT 
        date,
        AVG(CASE WHEN prev_momentum_rank <= prev_total_stocks * 0.10 
                 THEN daily_return END) AS avg_leader_return,
        AVG(CASE WHEN prev_momentum_rank > prev_total_stocks * 0.90 
                 THEN daily_return END) AS avg_laggard_return,
        COUNT(CASE WHEN prev_momentum_rank <= prev_total_stocks * 0.10 
                   THEN 1 END) AS n_leaders,
        COUNT(CASE WHEN prev_momentum_rank > prev_total_stocks * 0.90 
                   THEN 1 END) AS n_laggards
    FROM daily_returns
    GROUP BY date
)
SELECT 
    date,
    avg_leader_return,
    avg_laggard_return,
    n_leaders,
    n_laggards,
    CASE 
        WHEN avg_leader_return > avg_laggard_return THEN avg_leader_return + avg_laggard_return
        ELSE 0
    END AS dom_contribution
FROM leader_laggard_returns
WHERE avg_leader_return IS NOT NULL 
  AND avg_laggard_return IS NOT NULL
ORDER BY date;
