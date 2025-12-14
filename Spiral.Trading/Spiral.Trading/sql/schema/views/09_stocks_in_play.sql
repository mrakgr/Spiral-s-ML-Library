-- Stocks In Play: Top 10 stocks per day based on relative volume, gap, and liquidity
-- Table macro with parameterized filters for rvol, gap_pct, and minimum dollar volume
DROP MACRO IF EXISTS stocks_in_play;
CREATE MACRO stocks_in_play(
    min_rvol := 3,
    min_gap_pct := 0.05,
    min_avg_dollar_volume := 100000000
) AS TABLE
WITH daily_metrics AS (
    SELECT
        p.ticker,
        p.date,
        p.adj_open,
        p.adj_high,
        p.adj_low,
        p.adj_close,
        p.adj_volume,
        p_prev.adj_close AS prev_close,
        v.avg_dollar_volume_4w,
        -- Opening gap: % change from previous close to today's open
        (p.adj_open - p_prev.adj_close) / p_prev.adj_close AS gap_pct,
        -- Daily range as % of price
        (p.adj_high - p.adj_low) / p.adj_low AS range_pct,
        -- Today's dollar volume
        p.adj_close * p.adj_volume AS dollar_volume,
        -- Relative volume: today's dollar volume vs 4-week average
        (p.adj_close * p.adj_volume) / NULLIF(v.avg_dollar_volume_4w, 0) AS rvol
    FROM split_adjusted_prices p
    JOIN trading_calendar tc ON p.date = tc.current_date
    JOIN split_adjusted_prices p_prev
        ON p_prev.ticker = p.ticker
        AND p_prev.date = tc.date_prev
    JOIN stock_dollar_volume_4w v
        ON v.ticker = p.ticker
        AND v.date = p.date
    WHERE v.avg_dollar_volume_4w >= min_avg_dollar_volume
),
ranked AS (
    SELECT
        *,
        -- Composite score: weight RVOL and absolute gap
        (rvol * 0.5 + ABS(gap_pct) * 100 * 0.5) AS in_play_score,
        ROW_NUMBER() OVER (PARTITION BY date ORDER BY in_play_score DESC) AS rank
    FROM daily_metrics
    WHERE rvol >= min_rvol
      AND ABS(gap_pct) >= min_gap_pct
)
SELECT
    ticker,
    date,
    adj_open,
    adj_close,
    prev_close,
    gap_pct,
    range_pct,
    rvol,
    avg_dollar_volume_4w,
    in_play_score,
    rank
FROM ranked
WHERE rank <= 10
ORDER BY date, rank;
