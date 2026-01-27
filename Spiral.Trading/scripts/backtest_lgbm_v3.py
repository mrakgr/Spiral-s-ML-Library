#!/usr/bin/env python3
"""Backtest for LightGBM v3 with bar-relative normalization."""

import numpy as np
import lightgbm as lgb
from backtest_transformer_v3 import build_features_v3, TradingSystem, TREND_POSITION


def load_model(path='data/lgbm_v3_model.txt'):
    return lgb.Booster(model_file=path)


def build_lgbm_features(df, i, window_size=60):
    """Build flattened features for LightGBM."""
    f_1s, f_1m, f_5m = build_features_v3(df, i, window_size)
    
    # Flatten 1s
    f_1s_flat = f_1s.flatten()  # (180,)
    
    # Summary stats for 1m
    f_1m_stats = np.array([
        f_1m[:, 0].mean(), f_1m[:, 0].std(), f_1m[:, 0].min(), f_1m[:, 0].max(),
        f_1m[:, 1].mean(), f_1m[:, 1].std(), f_1m[:, 1].min(), f_1m[:, 1].max(),
        f_1m[:, 2].mean(), f_1m[:, 2].std(), f_1m[:, 2].min(), f_1m[:, 2].max(),
    ])
    
    # Summary stats for 5m
    f_5m_stats = np.array([
        f_5m[:, 0].mean(), f_5m[:, 0].std(), f_5m[:, 0].min(), f_5m[:, 0].max(),
        f_5m[:, 1].mean(), f_5m[:, 1].std(), f_5m[:, 1].min(), f_5m[:, 1].max(),
        f_5m[:, 2].mean(), f_5m[:, 2].std(), f_5m[:, 2].min(), f_5m[:, 2].max(),
    ])
    
    return np.concatenate([f_1s_flat, f_1m_stats, f_5m_stats])


def backtest_day(model, df, window_size=60, stride=5):
    system = TradingSystem(entry_threshold=0.90)
    prices = df['close'].values
    positions, pnls = [], []
    cumulative_pnl = 0.0
    
    for i in range(0, len(prices) - window_size, stride):
        features = build_lgbm_features(df, i, window_size)
        features = np.nan_to_num(features, nan=0.0).reshape(1, -1)
        probs = model.predict(features)[0]
        
        position = system.update(probs)
        positions.append(position)
        
        if i + window_size + stride < len(prices):
            price_change = prices[i + window_size + stride] - prices[i + window_size]
            cumulative_pnl += position * price_change
        pnls.append(cumulative_pnl)
    
    return {
        'final_pnl': cumulative_pnl,
        'num_trades': np.sum(np.abs(np.diff(positions)) > 0)
    }


def main():
    import pyarrow.parquet as pq
    
    print("Loading model...")
    model = load_model()
    
    print("Loading test data...")
    pf = pq.ParquetFile('data/test.parquet')
    
    day_indices = [0, 50, 100, 200, 500]
    all_pnls, all_trades = [], []
    
    for day_idx in day_indices:
        df = pf.read_row_group(day_idx).to_pandas()
        result = backtest_day(model, df)
        print(f"Day {day_idx}: PnL={result['final_pnl']:.2f}, Trades={result['num_trades']}")
        all_pnls.append(result['final_pnl'])
        all_trades.append(result['num_trades'])
    
    print(f"\nSummary over {len(day_indices)} days:")
    print(f"  Total PnL: {sum(all_pnls):.2f}")
    print(f"  Avg PnL/day: {np.mean(all_pnls):.2f}")
    print(f"  Avg trades/day: {np.mean(all_trades):.1f}")


if __name__ == "__main__":
    main()
