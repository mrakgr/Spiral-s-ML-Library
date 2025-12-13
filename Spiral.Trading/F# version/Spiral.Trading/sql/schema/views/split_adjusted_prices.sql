-- View for split-adjusted prices
-- Uses the materialized split_adjustment_factors table for adj_factor
DROP VIEW IF EXISTS split_adjusted_prices;
CREATE VIEW split_adjusted_prices AS
SELECT
    dp.ticker,
    dp.date,
    dp.open,
    dp.high,
    dp.low,
    dp.close,
    dp.volume,
    dp.transactions,
    saf.adj_factor,
    dp.open / saf.adj_factor AS adj_open,
    dp.high / saf.adj_factor AS adj_high,
    dp.low / saf.adj_factor AS adj_low,
    dp.close / saf.adj_factor AS adj_close,
    CAST(dp.volume * saf.adj_factor AS INTEGER) AS adj_volume
FROM daily_prices dp
JOIN split_adjustment_factors saf
    ON saf.ticker = dp.ticker
    AND saf.date = dp.date;
