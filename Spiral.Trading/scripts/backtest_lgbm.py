#!/usr/bin/env python3
"""Backtest using LightGBM model."""

import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import matplotlib.pyplot as plt
from dataset import TradingDataset

TREND_NAMES = [
    'StrongUptrend', 'MidUptrend', 'WeakUptrend', 'Consolidation',
    'WeakDowntrend', 'MidDowntrend', 'StrongDowntrend'
]

STRONG_UP, MID_UP, WEAK_UP = 0, 1, 2
CONSOLIDATION = 3
WEAK_DOWN, MID_DOWN, STRONG_DOWN = 4, 5, 6

TREND_POSITION = {
    STRONG_UP: 1, MID_UP: 1, WEAK_UP: 1,
    CONSOLIDATION: 0,
    WEAK_DOWN: -1, MID_DOWN: -1, STRONG_DOWN: -1
}


def extract_features(sample):
    """Extract flattened features from a dataset sample."""
    f_1s = sample['features_1s'].numpy().flatten()
    
    f_1m = sample['features_1m'].numpy()
    f_1m_stats = np.array([
        f_1m[:, 0].mean(), f_1m[:, 0].std(), f_1m[:, 0].min(), f_1m[:, 0].max(),
        f_1m[:, 1].mean(), f_1m[:, 1].std(), f_1m[:, 1].min(), f_1m[:, 1].max(),
        f_1m[:, 2].mean(), f_1m[:, 2].std(), f_1m[:, 2].min(), f_1m[:, 2].max(),
        f_1m[:, 3].mean(), f_1m[:, 3].std(), f_1m[:, 3].min(), f_1m[:, 3].max(),
    ])
    
    f_5m = sample['features_5m'].numpy()
    f_5m_stats = np.array([
        f_5m[:, 0].mean(), f_5m[:, 0].std(), f_5m[:, 0].min(), f_5m[:, 0].max(),
        f_5m[:, 1].mean(), f_5m[:, 1].std(), f_5m[:, 1].min(), f_5m[:, 1].max(),
        f_5m[:, 2].mean(), f_5m[:, 2].std(), f_5m[:, 2].min(), f_5m[:, 2].max(),
        f_5m[:, 3].mean(), f_5m[:, 3].std(), f_5m[:, 3].min(), f_5m[:, 3].max(),
    ])
    
    return np.concatenate([f_1s, f_1m_stats, f_5m_stats]).astype(np.float32)


class TradingSystem:
    def __init__(self, entry_threshold=0.90):
        self.entry_threshold = entry_threshold
        self.position = 0
        self.current_trend = None
    
    def update(self, probs):
        max_prob = probs.max()
        max_trend = probs.argmax()
        
        if max_prob >= self.entry_threshold and max_trend != self.current_trend:
            self.current_trend = max_trend
            self.position = TREND_POSITION[max_trend]
        
        return self.position


def train_model():
    """Train LightGBM model and return it."""
    from sklearn.metrics import accuracy_score
    
    print("Loading training data...")
    ds = TradingDataset('data/train.parquet', window_size=60, stride=5)
    n = 500_000
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)
    
    X, y = [], []
    for i, idx in enumerate(indices):
        sample = ds[idx]
        X.append(extract_features(sample))
        y.append(sample['trend'].item())
        if (i + 1) % 100000 == 0:
            print(f"  {i + 1:,}/{n:,}")
    
    X = np.nan_to_num(np.array(X, dtype=np.float32), nan=0.0)
    y = np.array(y, dtype=np.int32)
    
    print("Training LightGBM...")
    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'multiclass',
        'num_class': 7,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'verbose': -1,
    }
    model = lgb.train(params, train_data, num_boost_round=100)
    return model


def backtest_day(model, day_idx, pf, window_size=60, stride=5):
    """Backtest one day."""
    df = pf.read_row_group(day_idx).to_pandas()
    system = TradingSystem(entry_threshold=0.90)
    
    prices = df['close'].values
    positions, pnls, times = [], [], []
    cumulative_pnl = 0.0
    
    for i in range(0, len(prices) - window_size, stride):
        end = i + window_size
        pos = end - 1
        first_open = df['open'].iloc[i]
        
        # Build features manually (same as dataset)
        f_1s = np.stack([
            (df['open'].iloc[i:end].values - first_open) / first_open,
            (df['high'].iloc[i:end].values - first_open) / first_open,
            (df['low'].iloc[i:end].values - first_open) / first_open,
            (df['close'].iloc[i:end].values - first_open) / first_open,
        ], axis=1).flatten()
        
        # 1m stats
        all_1m = np.arange(59, pos + 1, 60)
        if len(all_1m) == 0 or all_1m[-1] != pos:
            all_1m = np.append(all_1m, pos)
        f_1m = np.stack([
            (df['open_1m_partial'].iloc[all_1m].values - first_open) / first_open,
            (df['high_1m_partial'].iloc[all_1m].values - first_open) / first_open,
            (df['low_1m_partial'].iloc[all_1m].values - first_open) / first_open,
            (df['close_1m_partial'].iloc[all_1m].values - first_open) / first_open,
        ], axis=1)
        f_1m_stats = np.array([
            f_1m[:, j].mean() for j in range(4) for _ in ['mean']
        ] + [
            f_1m[:, j].std() for j in range(4) for _ in ['std']
        ] + [
            f_1m[:, j].min() for j in range(4) for _ in ['min']
        ] + [
            f_1m[:, j].max() for j in range(4) for _ in ['max']
        ])
        # Reorder to match training
        f_1m_stats = np.array([
            f_1m[:, 0].mean(), f_1m[:, 0].std(), f_1m[:, 0].min(), f_1m[:, 0].max(),
            f_1m[:, 1].mean(), f_1m[:, 1].std(), f_1m[:, 1].min(), f_1m[:, 1].max(),
            f_1m[:, 2].mean(), f_1m[:, 2].std(), f_1m[:, 2].min(), f_1m[:, 2].max(),
            f_1m[:, 3].mean(), f_1m[:, 3].std(), f_1m[:, 3].min(), f_1m[:, 3].max(),
        ])
        
        # 5m stats
        all_5m = np.arange(299, pos + 1, 300)
        if len(all_5m) == 0 or all_5m[-1] != pos:
            all_5m = np.append(all_5m, pos)
        f_5m = np.stack([
            (df['open_5m_partial'].iloc[all_5m].values - first_open) / first_open,
            (df['high_5m_partial'].iloc[all_5m].values - first_open) / first_open,
            (df['low_5m_partial'].iloc[all_5m].values - first_open) / first_open,
            (df['close_5m_partial'].iloc[all_5m].values - first_open) / first_open,
        ], axis=1)
        f_5m_stats = np.array([
            f_5m[:, 0].mean(), f_5m[:, 0].std(), f_5m[:, 0].min(), f_5m[:, 0].max(),
            f_5m[:, 1].mean(), f_5m[:, 1].std(), f_5m[:, 1].min(), f_5m[:, 1].max(),
            f_5m[:, 2].mean(), f_5m[:, 2].std(), f_5m[:, 2].min(), f_5m[:, 2].max(),
            f_5m[:, 3].mean(), f_5m[:, 3].std(), f_5m[:, 3].min(), f_5m[:, 3].max(),
        ])
        
        features = np.concatenate([f_1s, f_1m_stats, f_5m_stats]).reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0).astype(np.float32)
        
        probs = model.predict(features)[0]
        position = system.update(probs)
        
        positions.append(position)
        times.append(end)
        
        if end + stride < len(prices):
            price_change = prices[end + stride] - prices[end]
            cumulative_pnl += position * price_change
        pnls.append(cumulative_pnl)
    
    return {
        'times': np.array(times),
        'positions': np.array(positions),
        'pnls': np.array(pnls),
        'prices': prices,
        'final_pnl': cumulative_pnl,
        'num_trades': np.sum(np.abs(np.diff(positions)) > 0) if len(positions) > 1 else 0
    }


def plot_backtest(result, day_idx, save_path):
    """Plot backtest results."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    times_min = result['times'] / 60
    prices = result['prices']
    pos = result['positions']
    
    # Price with position overlay
    ax1 = axes[0]
    ax1.plot(np.arange(len(prices)) / 60, prices, 'k-', linewidth=0.5, alpha=0.7)
    for i in range(len(times_min) - 1):
        color = 'green' if pos[i] == 1 else 'red' if pos[i] == -1 else 'gray'
        alpha = 0.3 if pos[i] != 0 else 0.1
        ax1.axvspan(times_min[i], times_min[i+1], color=color, alpha=alpha)
    ax1.set_ylabel('Price')
    ax1.set_title(f'Day {day_idx}: Price (green=long, red=short)')
    
    # Position
    ax2 = axes[1]
    ax2.step(times_min, pos, where='post', linewidth=1)
    ax2.set_ylabel('Position')
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title(f'Position (trades: {result["num_trades"]})')
    
    # PnL
    ax3 = axes[2]
    ax3.plot(times_min, result['pnls'], 'b-', linewidth=1)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Cumulative PnL')
    ax3.set_title(f'PnL: {result["final_pnl"]:.2f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def main():
    print("Training LightGBM model...")
    model = train_model()
    
    print("\nLoading test data...")
    pf = pq.ParquetFile('data/test.parquet')
    
    day_indices = [0, 50, 100, 200, 500]
    all_pnls = []
    all_trades = []
    
    for day_idx in day_indices:
        result = backtest_day(model, day_idx, pf)
        print(f"Day {day_idx}: PnL={result['final_pnl']:.2f}, Trades={result['num_trades']}")
        all_pnls.append(result['final_pnl'])
        all_trades.append(result['num_trades'])
        plot_backtest(result, day_idx, f'data/backtest_lgbm_day{day_idx}.png')
    
    print(f"\nSummary over {len(day_indices)} days:")
    print(f"  Total PnL: {sum(all_pnls):.2f}")
    print(f"  Avg PnL/day: {np.mean(all_pnls):.2f}")
    print(f"  Avg trades/day: {np.mean(all_trades):.1f}")


if __name__ == "__main__":
    main()
