#!/usr/bin/env python3
"""Backtest for MLP-Mixer v3 with bar-relative normalization."""

import torch
import numpy as np
from train_mixer_v3 import TradingMixerV3
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


def load_model(path='data/mixer_v3_model.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingMixerV3().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, device


def backtest_day(model, device, dataset, row_group):
    system = TradingSystem(entry_threshold=0.90)
    dataset._load_row_group(row_group)
    prices = dataset._cached_data['close']
    positions, pnls = [], []
    cumulative_pnl = 0.0
    
    model.eval()
    with torch.no_grad():
        for pos in range(0, len(prices)):
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
    
    return {
        'final_pnl': cumulative_pnl,
        'num_trades': np.sum(np.abs(np.diff(positions)) > 0)
    }


def main():
    print("Loading model...")
    model, device = load_model()
    
    print("Loading test data...")
    dataset = TradingDataset('data/test.parquet')
    
    day_indices = [0, 50, 100, 200, 500]
    all_pnls, all_trades = [], []
    
    for day_idx in day_indices:
        result = backtest_day(model, device, dataset, day_idx)
        print(f"Day {day_idx}: PnL={result['final_pnl']:.2f}, Trades={result['num_trades']}")
        all_pnls.append(result['final_pnl'])
        all_trades.append(result['num_trades'])
    
    print(f"\nSummary over {len(day_indices)} days:")
    print(f"  Total PnL: {sum(all_pnls):.2f}")
    print(f"  Avg PnL/day: {np.mean(all_pnls):.2f}")
    print(f"  Avg trades/day: {np.mean(all_trades):.1f}")


if __name__ == "__main__":
    main()
