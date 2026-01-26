#!/usr/bin/env python3
"""Backtest using MLP-Mixer model."""

import torch
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from train_mixer import TradingMixer

TREND_POSITION = {0: 1, 1: 1, 2: 1, 3: 0, 4: -1, 5: -1, 6: -1}


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


def build_features(df, i, window_size=60):
    """Build features for a single prediction."""
    end = i + window_size
    pos = end - 1
    first_open = df['open'].iloc[i]
    
    # 1s features
    f_1s = np.stack([
        (df['open'].iloc[i:end].values - first_open) / first_open,
        (df['high'].iloc[i:end].values - first_open) / first_open,
        (df['low'].iloc[i:end].values - first_open) / first_open,
        (df['close'].iloc[i:end].values - first_open) / first_open,
    ], axis=1).astype(np.float32)
    
    # 1m features
    all_1m = np.arange(59, pos + 1, 60)
    if len(all_1m) == 0 or all_1m[-1] != pos:
        all_1m = np.append(all_1m, pos)
    
    f_1m = np.zeros((60, 4), dtype=np.float32)
    n_1m = min(len(all_1m), 60)
    idx_1m = all_1m[-n_1m:]
    start_1m = 60 - n_1m
    f_1m[start_1m:, 0] = (df['open_1m_partial'].iloc[idx_1m].values - first_open) / first_open
    f_1m[start_1m:, 1] = (df['high_1m_partial'].iloc[idx_1m].values - first_open) / first_open
    f_1m[start_1m:, 2] = (df['low_1m_partial'].iloc[idx_1m].values - first_open) / first_open
    f_1m[start_1m:, 3] = (df['close_1m_partial'].iloc[idx_1m].values - first_open) / first_open
    
    # 5m features
    all_5m = np.arange(299, pos + 1, 300)
    if len(all_5m) == 0 or all_5m[-1] != pos:
        all_5m = np.append(all_5m, pos)
    
    f_5m = np.zeros((78, 4), dtype=np.float32)
    n_5m = min(len(all_5m), 78)
    idx_5m = all_5m[-n_5m:]
    start_5m = 78 - n_5m
    f_5m[start_5m:, 0] = (df['open_5m_partial'].iloc[idx_5m].values - first_open) / first_open
    f_5m[start_5m:, 1] = (df['high_5m_partial'].iloc[idx_5m].values - first_open) / first_open
    f_5m[start_5m:, 2] = (df['low_5m_partial'].iloc[idx_5m].values - first_open) / first_open
    f_5m[start_5m:, 3] = (df['close_5m_partial'].iloc[idx_5m].values - first_open) / first_open
    
    return f_1s, f_1m, f_5m


def backtest_day(model, device, day_idx, pf, window_size=60, stride=5):
    """Backtest one day."""
    df = pf.read_row_group(day_idx).to_pandas()
    system = TradingSystem(entry_threshold=0.90)
    
    prices = df['close'].values
    positions, pnls, times = [], [], []
    cumulative_pnl = 0.0
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prices) - window_size, stride):
            f_1s, f_1m, f_5m = build_features(df, i, window_size)
            
            x_1s = torch.from_numpy(f_1s).unsqueeze(0).to(device)
            x_1m = torch.from_numpy(f_1m).unsqueeze(0).to(device)
            x_5m = torch.from_numpy(f_5m).unsqueeze(0).to(device)
            
            _, trend_logits = model(x_1s, x_1m, x_5m)
            probs = torch.softmax(trend_logits, dim=1).cpu().numpy()[0]
            
            position = system.update(probs)
            positions.append(position)
            times.append(i + window_size)
            
            if i + window_size + stride < len(prices):
                price_change = prices[i + window_size + stride] - prices[i + window_size]
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
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    times_min = result['times'] / 60
    prices = result['prices']
    pos = result['positions']
    
    ax1 = axes[0]
    ax1.plot(np.arange(len(prices)) / 60, prices, 'k-', linewidth=0.5, alpha=0.7)
    for i in range(len(times_min) - 1):
        color = 'green' if pos[i] == 1 else 'red' if pos[i] == -1 else 'gray'
        ax1.axvspan(times_min[i], times_min[i+1], color=color, alpha=0.3 if pos[i] != 0 else 0.1)
    ax1.set_ylabel('Price')
    ax1.set_title(f'Day {day_idx}: Price (green=long, red=short)')
    
    ax2 = axes[1]
    ax2.step(times_min, pos, where='post')
    ax2.set_ylabel('Position')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title(f'Trades: {result["num_trades"]}')
    
    ax3 = axes[2]
    ax3.plot(times_min, result['pnls'], 'b-')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('PnL')
    ax3.set_title(f'PnL: {result["final_pnl"]:.2f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = TradingMixer().to(device)
    model.load_state_dict(torch.load('data/mixer_model.pt'))
    
    print("Loading test data...")
    pf = pq.ParquetFile('data/test.parquet')
    
    day_indices = [0, 50, 100, 200, 500]
    all_pnls = []
    all_trades = []
    
    for day_idx in day_indices:
        result = backtest_day(model, device, day_idx, pf)
        print(f"Day {day_idx}: PnL={result['final_pnl']:.2f}, Trades={result['num_trades']}")
        all_pnls.append(result['final_pnl'])
        all_trades.append(result['num_trades'])
        plot_backtest(result, day_idx, f'data/backtest_mixer_day{day_idx}.png')
    
    print(f"\nSummary over {len(day_indices)} days:")
    print(f"  Total PnL: {sum(all_pnls):.2f}")
    print(f"  Avg PnL/day: {np.mean(all_pnls):.2f}")
    print(f"  Avg trades/day: {np.mean(all_trades):.1f}")


if __name__ == "__main__":
    main()
