#!/usr/bin/env python3
"""Architecture-agnostic backtest script for trading models."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import TradingDataset

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


def load_model(model_class, weights_path, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def backtest_day(model, device, dataset, row_group):
    system = TradingSystem(entry_threshold=0.90)
    dataset._load_row_group(row_group)
    prices = dataset._cached_data['close']
    positions, pnls = [], []
    cumulative_pnl = 0.0
    
    model.eval()
    with torch.no_grad():
        for pos in range(len(prices)):
            sample = dataset.get_features_for(row_group, pos)
            x_1s = sample['features_1s'].unsqueeze(0).to(device)
            x_1m = sample['features_1m'].unsqueeze(0).to(device)
            x_5m = sample['features_5m'].unsqueeze(0).to(device)
            
            _, trend_logits = model(x_1s, x_1m, x_5m)
            probs = torch.softmax(trend_logits, dim=1).cpu().numpy()[0]
            position = system.update(probs)
            positions.append(position)
            
            if pos + 1 < len(prices):
                price_change = prices[pos + 1] - prices[pos]
                cumulative_pnl += position * price_change
            pnls.append(cumulative_pnl)
    
    positions = np.array(positions)
    return {
        'prices': prices,
        'positions': positions,
        'pnls': np.array(pnls),
        'final_pnl': cumulative_pnl,
        'num_trades': np.sum(np.abs(np.diff(positions)) > 0)
    }


def plot_backtest(result, day_idx, save_path):
    """Plot backtest results: price with positions, position changes, and cumulative PnL."""
    prices = result['prices']
    positions = result['positions']
    pnls = result['pnls']
    times = np.arange(len(prices)) / 60

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(times, prices, 'k-', linewidth=0.5, alpha=0.7)
    ax1.fill_between(times, prices.min(), prices.max(), where=positions == 1,
                     alpha=0.3, color='green', label='Long')
    ax1.fill_between(times, prices.min(), prices.max(), where=positions == -1,
                     alpha=0.3, color='red', label='Short')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Day {day_idx}: Price with Position Overlay (PnL={result["final_pnl"]:.2f}, Trades={result["num_trades"]})')
    ax1.legend(loc='upper left')

    ax2 = axes[1]
    ax2.step(times, positions, where='post', linewidth=1)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Position')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Flat', 'Long'])
    ax2.set_title('Position Changes')

    ax3 = axes[2]
    ax3.plot(times, pnls, 'b-', linewidth=1)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.fill_between(times, 0, pnls, where=pnls >= 0, alpha=0.3, color='green')
    ax3.fill_between(times, 0, pnls, where=pnls < 0, alpha=0.3, color='red')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Cumulative PnL')
    ax3.set_title('Cumulative PnL')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved backtest chart to {save_path}")
    plt.close()


def run_backtest(model_class, weights_path, model_name, day_indices=None):
    if day_indices is None:
        day_indices = [0, 50, 100, 200, 500]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading {model_name} model from {weights_path}...")
    model = load_model(model_class, weights_path, device)
    
    print("Loading test data...")
    dataset = TradingDataset('data/test.parquet')
    
    all_pnls, all_trades = [], []
    
    for day_idx in day_indices:
        result = backtest_day(model, device, dataset, day_idx)
        print(f"Day {day_idx}: PnL={result['final_pnl']:.2f}, Trades={result['num_trades']}")
        all_pnls.append(result['final_pnl'])
        all_trades.append(result['num_trades'])
        plot_backtest(result, day_idx, f'data/backtest_{model_name}_day{day_idx}.png')
    
    print(f"\nSummary over {len(day_indices)} days:")
    print(f"  Total PnL: {sum(all_pnls):.2f}")
    print(f"  Avg PnL/day: {np.mean(all_pnls):.2f}")
    print(f"  Avg trades/day: {np.mean(all_trades):.1f}")
    
    return {'pnls': all_pnls, 'trades': all_trades}


def main():
    parser = argparse.ArgumentParser(description='Backtest trading models')
    parser.add_argument('--model', type=str, required=True, choices=['mixer', 'gmlp'],
                        help='Model architecture to backtest')
    parser.add_argument('--weights', type=str, help='Path to model weights (optional)')
    parser.add_argument('--days', type=int, nargs='+', default=[0, 50, 100, 200, 500],
                        help='Day indices to backtest')
    args = parser.parse_args()
    
    if args.model == 'mixer':
        from train_mixer_v3 import TradingMixerV3
        model_class = TradingMixerV3
        weights_path = args.weights or 'data/mixer_v3_model.pt'
        model_name = 'mixer_v3'
    elif args.model == 'gmlp':
        from train_gmlp import TradingGMLP
        model_class = TradingGMLP
        weights_path = args.weights or 'data/gmlp_model.pt'
        model_name = 'gmlp'
    
    run_backtest(model_class, weights_path, model_name, args.days)


if __name__ == '__main__':
    main()
