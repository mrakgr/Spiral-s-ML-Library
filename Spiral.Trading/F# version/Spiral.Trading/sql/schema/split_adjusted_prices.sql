-- View for split-adjusted prices
-- Uses EXP(SUM(LN(split_ratio))) to calculate cumulative split factor
-- for all splits that occurred AFTER a given price date
CREATE VIEW IF NOT EXISTS split_adjusted_prices AS
WITH split_factors AS (
    SELECT 
        dp.ticker,
        dp.date,
        dp.open,
        dp.high,
        dp.low,
        dp.close,
        dp.volume,
        dp.transactions,
        COALESCE(
            (SELECT EXP(SUM(LN(s.split_ratio)))
             FROM splits s
             WHERE s.ticker = dp.ticker
             AND s.execution_date > dp.date),
            1.0
        ) AS adj_factor
    FROM daily_prices dp
)
SELECT
    ticker,
    date,
    open,
    high,
    low,
    close,
    volume,
    transactions,
    adj_factor,
    open / adj_factor AS adj_open,
    high / adj_factor AS adj_high,
    low / adj_factor AS adj_low,
    close / adj_factor AS adj_close,
    CAST(volume * adj_factor AS INTEGER) AS adj_volume
FROM split_factors;
