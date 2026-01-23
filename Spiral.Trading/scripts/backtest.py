#!/usr/bin/env python3
"""Simple trading system backtest based on trend predictions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import TradingDataset
from model import TradingLSTM

TREND_NAMES = [
    'StrongUptrend',
    'MidUptrend',
    'WeakUptrend',
    'Consolidation',
    'WeakDowntrend',
    'MidDowntrend',
    'StrongDowntrend'
]

# Trend indices
STRONG_UP, MID_UP, WEAK_UP = 0, 1, 2
CONSOLIDATION = 3
WEAK_DOWN, MID_DOWN, STRONG_DOWN = 4, 5, 6

# Position mapping for each trend
TREND_POSITION = {
    STRONG_UP: 1,    # Long
    MID_UP: 1,       # Long
    WEAK_UP: 1,      # Long
    CONSOLIDATION: 0, # Flat
    WEAK_DOWN: -1,   # Short
    MID_DOWN: -1,    # Short
    STRONG_DOWN: -1  # Short
}


def load_model(path='data/model.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingLSTM().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, device


def get_features(window_df):
    """Convert a window of OHLC data to model features."""
    first_open = window_df['open'].iloc[0]
    features = np.stack([
        (window_df['open'].values - first_open) / first_open,
        (window_df['high'].values - first_open) / first_open,
        (window_df['low'].values - first_open) / first_open,
        (window_df['close'].values - first_open) / first_open,
    ], axis=1).astype(np.float32)
    return features


class TradingSystem:
    """Trading system with 90% confidence threshold and hysteresis."""
    
    def __init__(self, entry_threshold=0.90):
        self.entry_threshold = entry_threshold
        self.position = 0  # -1 short, 0 flat, 1 long
        self.current_trend = None
    
    def update(self, probs):
        """Update position based on new probabilities. Returns new position."""
        max_prob = probs.max()
        max_trend = probs.argmax()
        
        # Only change position if we're 90% confident of a different trend
        if max_prob >= self.entry_threshold and max_trend != self.current_trend:
            self.current_trend = max_trend
            self.position = TREND_POSITION[max_trend]
        
        return self.position


def backtest_day(model, device, df, window_size=60, stride=5):
    """Backtest one day of data. Returns dict with results."""
    system = TradingSystem(entry_threshold=0.90)
    
    prices = df['close'].values
    true_trends = df['trend'].values
    
    positions = []
    pnls = []
    times = []
    pred_trends = []
    
    cumulative_pnl = 0.0
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prices) - window_size, stride):
            window = df.iloc[i:i + window_size]
            features = get_features(window)
            features_t = torch.from_numpy(features).unsqueeze(0).to(device)
            
            _, trend_logits = model(features_t)
            probs = torch.softmax(trend_logits, dim=1).cpu().numpy()[0]
            
            position = system.update(probs)
            positions.append(position)
            pred_trends.append(probs.argmax())
            times.append(i + window_size)
            
            # Calculate PnL from price change
            if i + window_size + stride < len(prices):
                price_change = prices[i + window_size + stride] - prices[i + window_size]
                pnl = position * price_change
                cumulative_pnl += pnl
                pnls.append(cumulative_pnl)
            else:
                pnls.append(cumulative_pnl)
    
    return {
        'times': np.array(times),
        'positions': np.array(positions),
        'pnls': np.array(pnls),
        'pred_trends': np.array(pred_trends),
        'prices': prices,
        'true_trends': true_trends,
        'final_pnl': cumulative_pnl,
        'num_trades': np.sum(np.abs(np.diff(positions)) > 0)
    }


def plot_backtest(result, day_idx, save_path):
    """Plot backtest results for one day."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    times_min = result['times'] / 60
    prices = result['prices']
    
    # Plot 1: Price with position overlay
    ax1 = axes[0]
    ax1.plot(np.arange(len(prices)) / 60, prices, 'k-', linewidth=0.5, alpha=0.7)
    
    # Color background by position
    pos = result['positions']
    for i in range(len(times_min) - 1):
        color = 'green' if pos[i] == 1 else 'red' if pos[i] == -1 else 'gray'
        alpha = 0.3 if pos[i] != 0 else 0.1
        ax1.axvspan(times_min[i], times_min[i+1], color=color, alpha=alpha)
    
    ax1.set_ylabel('Price')
    ax1.set_title(f'Day {day_idx}: Price with Position (green=long, red=short, gray=flat)')
    
    # Plot 2: Position over time
    ax2 = axes[1]
    ax2.step(times_min, pos, where='post', linewidth=1)
    ax2.set_ylabel('Position')
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title(f'Position (trades: {result["num_trades"]})')
    
    # Plot 3: Cumulative PnL
    ax3 = axes[2]
    ax3.plot(times_min, result['pnls'], 'b-', linewidth=1)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Cumulative PnL')
    ax3.set_title(f'Cumulative PnL: {result["final_pnl"]:.2f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def main():
    import pyarrow.parquet as pq
    
    print("Loading model...")
    model, device = load_model()
    
    print("Loading test data...")
    pf = pq.ParquetFile('data/test.parquet')
    
    # Backtest multiple days
    day_indices = [0, 50, 100, 200, 500]
    all_pnls = []
    all_trades = []
    
    for day_idx in day_indices:
        df = pf.read_row_group(day_idx).to_pandas()
        result = backtest_day(model, device, df)
        
        print(f"Day {day_idx}: PnL={result['final_pnl']:.2f}, Trades={result['num_trades']}")
        all_pnls.append(result['final_pnl'])
        all_trades.append(result['num_trades'])
        
        plot_backtest(result, day_idx, f'data/backtest_day{day_idx}.png')
    
    print(f"\nSummary over {len(day_indices)} days:")
    print(f"  Total PnL: {sum(all_pnls):.2f}")
    print(f"  Avg PnL/day: {np.mean(all_pnls):.2f}")
    print(f"  Avg trades/day: {np.mean(all_trades):.1f}")


if __name__ == "__main__":
    main()
