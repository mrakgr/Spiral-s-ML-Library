DROP VIEW IF EXISTS dom_indicator;
CREATE VIEW dom_indicator AS
WITH daily_returns AS (
    SELECT 
        r.ticker,
        r.date,
        r.adj_close,
        p_prev.adj_close AS adj_close_prev,
        GREATEST(-0.50, LEAST(2.00, 
            (r.adj_close - p_prev.adj_close) / p_prev.adj_close
        )) AS daily_return,
        r_prev.momentum_rank AS prev_momentum_rank,
        r_prev.total_stocks AS prev_total_stocks
    FROM stock_momentum_ranking r
    JOIN trading_calendar tc ON r.date = tc.current_date
    JOIN split_adjusted_prices p_prev 
        ON p_prev.ticker = r.ticker 
        AND p_prev.date = tc.date_prev
    JOIN stock_momentum_ranking r_prev
        ON r_prev.ticker = r.ticker
        AND r_prev.date = tc.date_prev
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
