"""
Janus Factor Market Timing Indicators
Based on "The Janus Factor" by Gary Edwin Anderson (2012)

This module implements the key market timing indicators:
1. RS (Relative Strength) - distance from Benchmark Equivalence Line
2. RSS (Relative Strength Spread) - gap between strongest and weakest groups
3. DOM (Direction of Momentum) - cumulative effect of positive feedback
4. RM (Relative Momentum) - relative strength of DOMs
5. CMA (Critical Moving Average) - adaptive moving average system

Key Concepts:
- Positive Feedback: Trend following behavior (buy strength, sell weakness)
- Negative Feedback: Contrarian behavior (buy weakness, sell strength)
- The universe expands during positive feedback, contracts during negative feedback
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional

class JanusFactorIndicators:
    """
    Calculate Janus Factor indicators for market timing.
    
    The core insight: Markets alternate between positive feedback (trending)
    and negative feedback (contrarian) regimes. These indicators detect the
    transitions.
    """
    
    def __init__(self, db_path: str = "data/trading.db"):
        """Initialize with database connection."""
        self.db_path = Path(db_path)
        self.conn = None
        
    def __enter__(self):
        """Context manager entry."""
        self.conn = sqlite3.connect(self.db_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            self.conn.close()
    
    def calculate_relative_strength(self, prices_df: pd.DataFrame, 
                                    lookback_days: int = 126) -> pd.DataFrame:
        """
        Calculate Relative Strength (RS) for each stock/group.
        
        RS measures the distance from the Benchmark Equivalence Line (BEL).
        Formula: RS = (Offense - Defense) / sqrt(2)
        
        Where:
        - Offense = % return over lookback period
        - Defense = volatility measure (could use std dev or drawdown)
        
        In Anderson's framework:
        - Positive RS = northwest of BEL (strong relative performance)
        - Negative RS = southeast of BEL (weak relative performance)
        
        Args:
            prices_df: DataFrame with columns [date, ticker/group, price]
            lookback_days: Period for RS calculation (default 126 ~= 6 months)
            
        Returns:
            DataFrame with RS values
        """
        print(f"\n[Calculating Relative Strength - {lookback_days} day lookback]")
        
        # Sort by ticker and date
        df = prices_df.copy().sort_values(['ticker', 'date'])
        
        # Calculate returns over lookback period
        df['price_lag'] = df.groupby('ticker')['price'].shift(lookback_days)
        df['offense'] = ((df['price'] / df['price_lag']) - 1) * 100  # % return
        
        # Calculate defense (volatility) - using rolling std of daily returns
        df['daily_return'] = df.groupby('ticker')['price'].pct_change()
        df['defense'] = df.groupby('ticker')['daily_return'].transform(
            lambda x: x.rolling(lookback_days, min_periods=lookback_days//2).std() * np.sqrt(252) * 100
        )
        
        # Calculate RS = (Offense - Defense) / sqrt(2)
        df['rs'] = (df['offense'] - df['defense']) / np.sqrt(2)
        
        # Calculate benchmark (market average) RS for each date
        benchmark_rs = df.groupby('date')['rs'].mean().rename('benchmark_rs')
        df = df.merge(benchmark_rs, on='date', how='left')
        
        # Relative RS (vs benchmark)
        df['relative_rs'] = df['rs'] - df['benchmark_rs']
        
        return df[['date', 'ticker', 'rs', 'relative_rs', 'offense', 'defense']].dropna()
    
    def calculate_rs_spread(self, rs_df: pd.DataFrame, 
                           top_pct: float = 0.10,
                           bottom_pct: float = 0.10) -> pd.DataFrame:
        """
        Calculate Relative Strength Spread (RSS).
        
        RSS = Average RS of top X% - Average RS of bottom X%
        
        A rising RSS indicates positive feedback (universe expanding).
        A falling RSS indicates negative feedback (universe contracting).
        
        Args:
            rs_df: DataFrame with RS values
            top_pct: Percentage for leaders (default 10%)
            bottom_pct: Percentage for laggards (default 10%)
            
        Returns:
            DataFrame with RSS by date
        """
        print(f"\n[Calculating RS Spread - top/bottom {top_pct*100}%]")
        
        results = []
        
        for date, group in rs_df.groupby('date'):
            # Sort by relative RS
            sorted_group = group.sort_values('relative_rs', ascending=False)
            n = len(sorted_group)
            
            # Get top and bottom percentiles
            n_top = max(1, int(n * top_pct))
            n_bottom = max(1, int(n * bottom_pct))
            
            leaders = sorted_group.head(n_top)
            laggards = sorted_group.tail(n_bottom)
            
            # Calculate averages
            avg_leader_rs = leaders['relative_rs'].mean()
            avg_laggard_rs = laggards['relative_rs'].mean()
            
            # Calculate spread
            rss = avg_leader_rs - avg_laggard_rs
            
            results.append({
                'date': date,
                'rss': rss,
                'avg_leader_rs': avg_leader_rs,
                'avg_laggard_rs': avg_laggard_rs,
                'n_stocks': n
            })
        
        return pd.DataFrame(results)
    
    def calculate_dom(self, prices_df: pd.DataFrame, 
                      rs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Direction of Momentum (DOM).
        
        DOM integrates price change with positive feedback.
        It's a cumulative measure that rises when capital flows from weak to strong
        (positive feedback) and falls when capital flows from strong to weak.
        
        DOM = Cumulative sum of (Daily Return * RS_sign)
        
        Where RS_sign indicates if the stock/group is strong (1) or weak (-1).
        
        The book suggests DOM is calculated as:
        - Sum of daily price changes, weighted by relative strength position
        - Essentially tracking momentum-driven capital flows
        
        Args:
            prices_df: DataFrame with prices
            rs_df: DataFrame with RS values
            
        Returns:
            DataFrame with DOM values
        """
        print("\n[Calculating Direction of Momentum (DOM)]")
        
        # Merge prices with RS
        df = prices_df.merge(
            rs_df[['date', 'ticker', 'relative_rs']], 
            on=['date', 'ticker'], 
            how='left'
        )
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date'])
        
        # Calculate daily returns
        df['daily_return'] = df.groupby('ticker')['price'].pct_change()
        
        # Determine if strong (above benchmark) or weak
        df['rs_sign'] = np.where(df['relative_rs'] > 0, 1, -1)
        
        # Weighted return (positive feedback contribution)
        df['weighted_return'] = df['daily_return'] * df['rs_sign']
        
        # Calculate DOM as cumulative sum of weighted returns
        df['dom'] = df.groupby('ticker')['weighted_return'].cumsum() * 100  # Scale to percentage
        
        # Calculate average DOM across all stocks for market-level indicator
        market_dom = df.groupby('date').agg({
            'weighted_return': 'mean',
            'daily_return': 'mean'
        }).reset_index()
        market_dom['market_dom'] = market_dom['weighted_return'].cumsum() * 100
        
        return df[['date', 'ticker', 'dom', 'daily_return']], market_dom
    
    def calculate_relative_momentum(self, dom_df: pd.DataFrame, 
                                    lookback_days: int = 126) -> pd.DataFrame:
        """
        Calculate Relative Momentum (RM).
        
        RM substitutes DOM for price in the RS calculation.
        
        Instead of comparing price performance, RM compares the momentum
        (DOM) performance of different stocks/groups.
        
        Formula: RM = RS of DOMs (not of prices)
        
        Args:
            dom_df: DataFrame with DOM values
            lookback_days: Period for RM calculation (default 126)
            
        Returns:
            DataFrame with RM values
        """
        print(f"\n[Calculating Relative Momentum - {lookback_days} day lookback]")
        
        # Sort by ticker and date
        df = dom_df.copy().sort_values(['ticker', 'date'])
        
        # Calculate DOM change over lookback period
        df['dom_lag'] = df.groupby('ticker')['dom'].shift(lookback_days)
        df['dom_change'] = df['dom'] - df['dom_lag']
        
        # Calculate 6-month sum of daily DOM changes (as mentioned in the book)
        df['dom_sum'] = df.groupby('ticker')['dom'].transform(
            lambda x: x.diff().rolling(lookback_days, min_periods=lookback_days//2).sum()
        )
        
        # Calculate benchmark (average) RM
        benchmark_rm = df.groupby('date')['dom_sum'].mean().rename('benchmark_rm')
        df = df.merge(benchmark_rm, on='date', how='left')
        
        # Relative RM
        df['relative_rm'] = df['dom_sum'] - df['benchmark_rm']
        
        return df[['date', 'ticker', 'dom_sum', 'relative_rm']].dropna()
    
    def calculate_rm_spread(self, rm_df: pd.DataFrame,
                           top_pct: float = 0.10,
                           bottom_pct: float = 0.10) -> pd.DataFrame:
        """
        Calculate Relative Momentum Spread.
        
        Similar to RSS but for momentum instead of strength.
        
        Args:
            rm_df: DataFrame with RM values
            top_pct: Percentage for leaders
            bottom_pct: Percentage for laggards
            
        Returns:
            DataFrame with RM spread by date
        """
        print(f"\n[Calculating RM Spread - top/bottom {top_pct*100}%]")
        
        results = []
        
        for date, group in rm_df.groupby('date'):
            sorted_group = group.sort_values('relative_rm', ascending=False)
            n = len(sorted_group)
            
            n_top = max(1, int(n * top_pct))
            n_bottom = max(1, int(n * bottom_pct))
            
            leaders = sorted_group.head(n_top)
            laggards = sorted_group.tail(n_bottom)
            
            avg_leader_rm = leaders['relative_rm'].mean()
            avg_laggard_rm = laggards['relative_rm'].mean()
            
            rm_spread = avg_leader_rm - avg_laggard_rm
            
            results.append({
                'date': date,
                'rm_spread': rm_spread,
                'avg_leader_rm': avg_leader_rm,
                'avg_laggard_rm': avg_laggard_rm
            })
        
        return pd.DataFrame(results)
    
    def calculate_cma(self, series: pd.Series, 
                      min_period: int = 29,
                      max_period: int = 60,
                      test_period: int = 126) -> Tuple[int, pd.Series]:
        """
        Calculate Critical Moving Average (CMA).
        
        An adaptive system that finds the optimal EMA lookback period
        by testing which period would have been most profitable over the
        recent test period.
        
        The CMA is "smart" - it adapts to changing market conditions by
        continuously re-optimizing the lookback period.
        
        Args:
            series: Time series to calculate CMA for (e.g., DOM)
            min_period: Minimum EMA period to test
            max_period: Maximum EMA period to test
            test_period: Number of days to test profitability over
            
        Returns:
            Tuple of (optimal_period, cma_series)
        """
        print(f"\n[Calculating Critical Moving Average - testing periods {min_period}-{max_period}]")
        
        best_period = min_period
        best_profit = -np.inf
        
        # Test each EMA period
        for period in range(min_period, max_period + 1):
            # Calculate EMA
            ema = series.ewm(span=period, adjust=False).mean()
            
            # Calculate signals: 1 if EMA rising, -1 if falling
            ema_change = ema.diff()
            signals = np.where(ema_change > 0, 1, -1)
            
            # Calculate returns from following signals
            returns = series.pct_change().shift(-1)  # Next period return
            strategy_returns = returns * signals
            
            # Calculate profit over test period
            recent_profit = strategy_returns.tail(test_period).sum()
            
            if recent_profit > best_profit:
                best_profit = recent_profit
                best_period = period
        
        # Calculate CMA with best period
        cma = series.ewm(span=best_period, adjust=False).mean()
        
        print(f"  Optimal period: {best_period} days (profit: {best_profit:.4f})")
        
        return best_period, cma
    
    def generate_signals(self, market_dom: pd.DataFrame,
                        rm_spread: pd.DataFrame,
                        use_defensive: bool = True) -> pd.DataFrame:
        """
        Generate trading signals based on Janus Factor indicators.
        
        Strategy from the book:
        
        Long Strategy:
        - Simple: Hold longs when DOM CMA is rising
        - Defensive: Hold longs when BOTH DOM CMA AND RM Spread CMA are rising
        
        Short Strategy:
        - Hold shorts when DOM CMA is falling AND RM Spread CMA is rising
        - Exit to cash when RM Spread CMA falls (contrarian rally risk)
        
        Args:
            market_dom: DataFrame with market-level DOM
            rm_spread: DataFrame with RM spread
            use_defensive: If True, use defensive strategy (default True)
            
        Returns:
            DataFrame with trading signals
        """
        print("\n[Generating Trading Signals]")
        print(f"  Strategy: {'Defensive' if use_defensive else 'DOM-Only'}")
        
        # Merge DOM and RM spread
        signals = market_dom.merge(rm_spread[['date', 'rm_spread']], on='date', how='inner')
        
        # Calculate CMAs
        _, dom_cma = self.calculate_cma(signals['market_dom'])
        _, rm_spread_cma = self.calculate_cma(signals['rm_spread'])
        
        signals['dom_cma'] = dom_cma
        signals['rm_spread_cma'] = rm_spread_cma
        
        # Calculate CMA directions
        signals['dom_rising'] = signals['dom_cma'].diff() > 0
        signals['rm_spread_rising'] = signals['rm_spread_cma'].diff() > 0
        
        # Generate position signals
        if use_defensive:
            # Defensive long: both DOM and RM spread rising
            signals['position'] = np.where(
                signals['dom_rising'] & signals['rm_spread_rising'], 
                1,  # Long
                np.where(
                    ~signals['dom_rising'] & signals['rm_spread_rising'],
                    -1,  # Short
                    0  # Cash
                )
            )
        else:
            # DOM-only: long when DOM rising, cash otherwise
            signals['position'] = np.where(signals['dom_rising'], 1, 0)
        
        # Add position labels
        signals['signal'] = signals['position'].map({
            1: 'LONG',
            -1: 'SHORT',
            0: 'CASH'
        })
        
        return signals[['date', 'market_dom', 'dom_cma', 'rm_spread', 'rm_spread_cma', 
                       'dom_rising', 'rm_spread_rising', 'position', 'signal']]


def main():
    """Example usage of Janus Factor indicators."""
    
    print("=" * 80)
    print("JANUS FACTOR MARKET TIMING INDICATORS")
    print("Based on 'The Janus Factor' by Gary Edwin Anderson (2012)")
    print("=" * 80)
    
    # Initialize
    db_path = Path("data/trading.db")
    
    if not db_path.exists():
        print(f"\nError: Database not found at {db_path}")
        print("Please run the data download script first.")
        return
    
    with JanusFactorIndicators(db_path) as jf:
        # Load price data from database
        print("\n[Loading price data from database]")
        
        query = """
        SELECT 
            date,
            ticker,
            close as price,
            volume
        FROM daily_prices
        WHERE date >= date('now', '-2 years')
        ORDER BY ticker, date
        """
        
        prices_df = pd.read_sql_query(query, jf.conn)
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        print(f"  Loaded {len(prices_df):,} price records")
        print(f"  Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")
        print(f"  Number of tickers: {prices_df['ticker'].nunique()}")
        
        # Calculate indicators
        print("\n" + "=" * 80)
        print("CALCULATING INDICATORS")
        print("=" * 80)
        
        # 1. Relative Strength
        rs_df = jf.calculate_relative_strength(prices_df, lookback_days=126)
        print(f"  Calculated RS for {len(rs_df):,} records")
        
        # 2. RS Spread
        rss_df = jf.calculate_rs_spread(rs_df, top_pct=0.08, bottom_pct=0.08)
        print(f"  Calculated RSS for {len(rss_df)} trading days")
        
        # 3. Direction of Momentum
        dom_df, market_dom = jf.calculate_dom(prices_df, rs_df)
        print(f"  Calculated DOM for {len(dom_df):,} records")
        
        # 4. Relative Momentum
        rm_df = jf.calculate_relative_momentum(dom_df, lookback_days=126)
        print(f"  Calculated RM for {len(rm_df):,} records")
        
        # 5. RM Spread
        rm_spread_df = jf.calculate_rm_spread(rm_df, top_pct=0.08, bottom_pct=0.08)
        print(f"  Calculated RM Spread for {len(rm_spread_df)} trading days")
        
        # 6. Generate trading signals
        signals_df = jf.generate_signals(market_dom, rm_spread_df, use_defensive=True)
        print(f"  Generated signals for {len(signals_df)} trading days")
        
        # Display current signal
        print("\n" + "=" * 80)
        print("CURRENT MARKET SIGNAL")
        print("=" * 80)
        
        latest = signals_df.iloc[-1]
        print(f"\nDate: {latest['date']}")
        print(f"Signal: {latest['signal']}")
        print(f"\nDOM CMA: {'RISING' if latest['dom_rising'] else 'FALLING'}")
        print(f"RM Spread CMA: {'RISING' if latest['rm_spread_rising'] else 'FALLING'}")
        print(f"\nMarket DOM: {latest['market_dom']:.2f}")
        print(f"RM Spread: {latest['rm_spread']:.2f}")
        
        # Show signal distribution
        print("\n" + "=" * 80)
        print("SIGNAL DISTRIBUTION (Last 252 Days)")
        print("=" * 80)
        
        recent_signals = signals_df.tail(252)
        signal_counts = recent_signals['signal'].value_counts()
        
        for signal, count in signal_counts.items():
            pct = count / len(recent_signals) * 100
            print(f"{signal:6s}: {count:3d} days ({pct:5.1f}%)")
        
        print("\n" + "=" * 80)
        print("DONE")
        print("=" * 80)


if __name__ == "__main__":
    main()
