"""
Janus Factor Market Timing Indicators
Based on "The Janus Factor" by Gary Edwin Anderson (2012)

This module implements:
1. DOM (Direction of Momentum) - Tracks market expansion/contraction
   - Only accumulates when leaders (top 10% by 6mo momentum) outperform laggards (bottom 10%)
   - Flatlines when spread contracts
   
2. Performance Spread - Short-term timing indicator
   - Difference between avg 6mo momentum of leaders vs laggards
   
Trading signals:
- Best longs: Both DOM ↑ and Performance Spread ↑
- Best shorts: Both DOM ↓ and Performance Spread ↓

Stock universe filters:
- Minimum 6-month average price: $5
- Minimum 1-month average dollar volume: $100M
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class JanusFactorIndicators:
    """
    Calculate Janus Factor indicators using SQL for performance.
    
    Filters:
    - Min 6-month average price: $5
    - Min 1-month average dollar volume: $100M
    - Top/bottom 10% by 6-month momentum
    """
    
    def __init__(self, db_path: str = "data/trading.db"):
        """Initialize with database connection."""
        self.db_path = Path(db_path)
        self.conn = None
        
        # Configuration
        self.MOMENTUM_DAYS = 126  # 6 months
        self.MIN_AVG_PRICE = 5.0
        self.MIN_DOLLAR_VOLUME = 100_000_000  # $100M per month
        self.PRICE_AVG_DAYS = 126  # 6 months
        self.VOLUME_AVG_DAYS = 21  # 1 month
        self.LEADER_PCT = 0.10  # Top 10%
        self.LAGGARD_PCT = 0.10  # Bottom 10%
        
    def __enter__(self):
        """Context manager entry."""
        self.conn = sqlite3.connect(self.db_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            self.conn.close()
    
    def _create_filtered_universe(self):
        """Create filtered stock universe with momentum and eligibility criteria."""
        print("\n[Creating filtered stock universe]")
        
        # Drop existing views if they exist
        self.conn.execute("DROP VIEW IF EXISTS eligible_stocks")
        
        query = f"""
        CREATE VIEW eligible_stocks AS
        WITH split_adjusted AS (
            -- Calculate split-adjusted prices
            SELECT 
                dp.ticker,
                dp.date,
                dp.close,
                dp.volume,
                COALESCE(
                    (SELECT EXP(SUM(LOG(s.split_ratio)))
                     FROM splits s
                     WHERE s.ticker = dp.ticker
                     AND s.execution_date > dp.date),
                    1.0
                ) AS adj_factor
            FROM daily_prices dp
        ),
        adjusted_prices AS (
            SELECT
                ticker,
                date,
                close / adj_factor AS adj_close,
                CAST(volume * adj_factor AS INTEGER) AS adj_volume,
                close * volume AS dollar_volume
            FROM split_adjusted
        ),
        lagged_data AS (
            SELECT
                ticker,
                date,
                adj_close,
                adj_volume,
                dollar_volume,
                LAG(adj_close, {self.MOMENTUM_DAYS}) OVER (PARTITION BY ticker ORDER BY date) AS price_lag,
                LAG(date, {self.MOMENTUM_DAYS}) OVER (PARTITION BY ticker ORDER BY date) AS date_lag
            FROM adjusted_prices
        ),
        momentum_calc AS (
            SELECT
                ticker,
                date,
                adj_close,
                adj_volume,
                dollar_volume,
                price_lag,
                CASE
                    WHEN price_lag IS NOT NULL AND price_lag > 0
                    THEN (adj_close - price_lag) / price_lag
                    ELSE NULL
                END AS momentum_6m
            FROM lagged_data
        ),
        rolling_averages AS (
            SELECT
                mc.ticker,
                mc.date,
                mc.adj_close,
                mc.momentum_6m,
                AVG(mc.adj_close) OVER (
                    PARTITION BY mc.ticker 
                    ORDER BY mc.date 
                    ROWS BETWEEN {self.PRICE_AVG_DAYS - 1} PRECEDING AND CURRENT ROW
                ) AS avg_price_6m,
                COUNT(mc.adj_close) OVER (
                    PARTITION BY mc.ticker 
                    ORDER BY mc.date 
                    ROWS BETWEEN {self.PRICE_AVG_DAYS - 1} PRECEDING AND CURRENT ROW
                ) AS price_count,
                AVG(mc.dollar_volume) OVER (
                    PARTITION BY mc.ticker 
                    ORDER BY mc.date 
                    ROWS BETWEEN {self.VOLUME_AVG_DAYS - 1} PRECEDING AND CURRENT ROW
                ) AS avg_dollar_volume_1m,
                COUNT(mc.dollar_volume) OVER (
                    PARTITION BY mc.ticker 
                    ORDER BY mc.date 
                    ROWS BETWEEN {self.VOLUME_AVG_DAYS - 1} PRECEDING AND CURRENT ROW
                ) AS volume_count
            FROM momentum_calc mc
        )
        SELECT
            ticker,
            date,
            adj_close,
            momentum_6m,
            avg_price_6m,
            avg_dollar_volume_1m
        FROM rolling_averages
        WHERE momentum_6m IS NOT NULL
        AND price_count >= {self.PRICE_AVG_DAYS}
        AND volume_count >= {self.VOLUME_AVG_DAYS}
        AND avg_price_6m >= {self.MIN_AVG_PRICE}
        AND avg_dollar_volume_1m >= {self.MIN_DOLLAR_VOLUME}
        """
        
        self.conn.execute(query)
        print("✓ Created eligible_stocks view")
    
    def calculate_dom_and_performance_spread(self) -> pd.DataFrame:
        """
        Calculate DOM and Performance Spread indicators.
        
        Steps:
        1. For each date, identify top 10% (leaders) and bottom 10% (laggards) by 6mo momentum
        2. Calculate daily returns for each stock
        3. Calculate average daily return for leaders and laggards
        4. DOM: Cumulative sum of (leader_avg_return - laggard_avg_return) when positive, else 0
        5. Performance Spread: Average 6mo momentum of leaders - average 6mo momentum of laggards
        
        Returns:
            DataFrame with columns: date, dom, performance_spread, n_leaders, n_laggards
        """
        print("\n[Calculating DOM and Performance Spread]")
        
        query = f"""
        WITH daily_data AS (
            SELECT
                ticker,
                date,
                adj_close,
                momentum_6m,
                adj_close / LAG(adj_close, 1) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return
            FROM eligible_stocks
        ),
        filtered_data AS (
            -- Filter out extreme daily returns (likely data errors)
            -- Exclude returns > 100% or < -90% which are almost certainly bad data
            SELECT
                ticker,
                date,
                adj_close,
                momentum_6m,
                daily_return
            FROM daily_data
            WHERE momentum_6m IS NOT NULL
            AND daily_return IS NOT NULL
            AND daily_return <= 1.0   -- Max +100% daily return
            AND daily_return >= -0.9  -- Max -90% daily return
        ),
        -- Rank stocks based on PREVIOUS day's momentum to avoid look-ahead bias
        previous_day_rankings AS (
            SELECT
                ticker,
                date,
                adj_close,
                momentum_6m,
                daily_return,
                LAG(momentum_6m, 1) OVER (PARTITION BY ticker ORDER BY date) AS prev_momentum,
                COUNT(*) OVER (PARTITION BY date) AS n_stocks
            FROM filtered_data
        ),
        ranked_stocks AS (
            SELECT
                ticker,
                date,
                adj_close,
                momentum_6m,
                prev_momentum,
                daily_return,
                n_stocks,
                -- Rank based on PREVIOUS day's momentum
                RANK() OVER (PARTITION BY date ORDER BY prev_momentum DESC) AS momentum_rank
            FROM previous_day_rankings
            WHERE prev_momentum IS NOT NULL  -- Need previous day's momentum
        ),
        leaders_laggards AS (
            SELECT
                date,
                n_stocks,
                CAST(n_stocks * {self.LEADER_PCT} AS INTEGER) AS n_leaders_target,
                CAST(n_stocks * {self.LAGGARD_PCT} AS INTEGER) AS n_laggards_target,
                -- Leaders (top 10% by PREVIOUS day's momentum)
                AVG(CASE WHEN momentum_rank <= CAST(n_stocks * {self.LEADER_PCT} AS INTEGER) 
                         THEN daily_return END) AS avg_leader_return,
                AVG(CASE WHEN momentum_rank <= CAST(n_stocks * {self.LEADER_PCT} AS INTEGER) 
                         THEN momentum_6m END) AS avg_leader_momentum,
                COUNT(CASE WHEN momentum_rank <= CAST(n_stocks * {self.LEADER_PCT} AS INTEGER) 
                           THEN 1 END) AS n_leaders,
                -- Laggards (bottom 10% by PREVIOUS day's momentum)
                AVG(CASE WHEN momentum_rank > n_stocks - CAST(n_stocks * {self.LAGGARD_PCT} AS INTEGER)
                         THEN daily_return END) AS avg_laggard_return,
                AVG(CASE WHEN momentum_rank > n_stocks - CAST(n_stocks * {self.LAGGARD_PCT} AS INTEGER)
                         THEN momentum_6m END) AS avg_laggard_momentum,
                COUNT(CASE WHEN momentum_rank > n_stocks - CAST(n_stocks * {self.LAGGARD_PCT} AS INTEGER)
                           THEN 1 END) AS n_laggards
            FROM ranked_stocks
            GROUP BY date, n_stocks
        )
        SELECT
            date,
            n_stocks,
            n_leaders,
            n_laggards,
            avg_leader_return,
            avg_laggard_return,
            avg_leader_momentum,
            avg_laggard_momentum,
            -- DOM: Sum of leaders + laggards when expanding, zero when contracting
            -- Expanding = leaders outperform laggards (positive feedback)
            -- This allows DOM to rise in bull markets and fall/flatline in bear markets
            CASE 
                WHEN avg_leader_return > avg_laggard_return 
                THEN avg_leader_return + avg_laggard_return
                ELSE 0
            END AS dom_contribution,
            -- Performance spread (for short-term timing)
            avg_leader_momentum - avg_laggard_momentum AS performance_spread
        FROM leaders_laggards
        WHERE avg_leader_return IS NOT NULL 
        AND avg_laggard_return IS NOT NULL
        ORDER BY date
        """
        
        df = pd.read_sql_query(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate cumulative DOM
        df['dom'] = (df['dom_contribution'] * 100).cumsum()  # Scale to percentage
        df['performance_spread'] = df['performance_spread'] * 100  # Scale to percentage
        
        print(f"  Calculated {len(df)} trading days")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Average stocks per day: {df['n_stocks'].mean():.0f}")
        print(f"  Average leaders/laggards: {df['n_leaders'].mean():.0f}/{df['n_laggards'].mean():.0f}")
        
        return df
    
    def calculate_indicators(self) -> pd.DataFrame:
        """
        Main entry point to calculate all indicators.
        
        Returns:
            DataFrame with DOM and Performance Spread indicators
        """
        print("=" * 80)
        print("JANUS FACTOR INDICATORS")
        print("=" * 80)
        
        # Create filtered universe
        self._create_filtered_universe()
        
        # Calculate indicators
        indicators_df = self.calculate_dom_and_performance_spread()
        
        # Add trend indicators using 1-month rate of change (21 trading days)
        # This smooths out daily noise and gives more stable signals
        lookback_days = 21
        indicators_df['dom_change_1m'] = indicators_df['dom'].diff(lookback_days)
        indicators_df['perf_spread_change_1m'] = indicators_df['performance_spread'].diff(lookback_days)
        indicators_df['dom_rising'] = indicators_df['dom_change_1m'] > 0
        indicators_df['perf_spread_rising'] = indicators_df['perf_spread_change_1m'] > 0
        
        # Generate signals
        indicators_df['signal'] = 'NEUTRAL'
        indicators_df.loc[
            indicators_df['dom_rising'] & indicators_df['perf_spread_rising'], 
            'signal'
        ] = 'LONG'
        indicators_df.loc[
            ~indicators_df['dom_rising'] & ~indicators_df['perf_spread_rising'], 
            'signal'
        ] = 'SHORT'
        
        return indicators_df


def main():
    """Example usage."""
    
    db_path = Path("data/trading.db")
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Please run data download script first.")
        return
    
    with JanusFactorIndicators(db_path) as jf:
        # Calculate indicators
        indicators_df = jf.calculate_indicators()
        
        # Display current status
        print("\n" + "=" * 80)
        print("CURRENT MARKET STATUS")
        print("=" * 80)
        
        latest = indicators_df.iloc[-1]
        print(f"\nDate: {latest['date'].strftime('%Y-%m-%d')}")
        print(f"Signal: {latest['signal']}")
        print(f"\nDOM: {latest['dom']:.2f} ({'RISING' if latest['dom_rising'] else 'FALLING'})")
        print(f"Performance Spread: {latest['performance_spread']:.2f}% ({'RISING' if latest['perf_spread_rising'] else 'FALLING'})")
        print(f"\nUniverse size: {latest['n_stocks']:.0f} stocks")
        print(f"Leaders: {latest['n_leaders']:.0f}, Laggards: {latest['n_laggards']:.0f}")
        
        # Show recent history
        print("\n" + "=" * 80)
        print("RECENT HISTORY (Last 10 Days)")
        print("=" * 80)
        
        recent = indicators_df.tail(10)[['date', 'dom', 'performance_spread', 'signal']]
        recent['date'] = recent['date'].dt.strftime('%Y-%m-%d')
        recent['dom'] = recent['dom'].round(2)
        recent['performance_spread'] = recent['performance_spread'].round(2)
        print(recent.to_string(index=False))
        
        # Signal distribution
        print("\n" + "=" * 80)
        print("SIGNAL DISTRIBUTION (Last 252 Days)")
        print("=" * 80)
        
        recent_signals = indicators_df.tail(252)
        signal_counts = recent_signals['signal'].value_counts()
        for signal, count in signal_counts.items():
            pct = count / len(recent_signals) * 100
            print(f"{signal:8s}: {count:3d} days ({pct:5.1f}%)")
        
        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)


if __name__ == "__main__":
    main()
