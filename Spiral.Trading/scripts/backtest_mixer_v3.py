#!/usr/bin/env python3
"""Backtest for MLP-Mixer v3 with bar-relative normalization."""

import torch
import numpy as np
from train_mixer_v3 import TradingMixerV3
from backtest_transformer_v3 import build_features_v3, TradingSystem, TREND_POSITION

def load_model(path='data/mixer_v3_model.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingMixerV3().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, device


def backtest_day(model, device, df, window_size=60, stride=5):
    system = TradingSystem(entry_threshold=0.90)
    prices = df['close'].values
    positions, pnls = [], []
    cumulative_pnl = 0.0
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prices) - window_size, stride):
            f_1s, f_1m, f_5m = build_features_v3(df, i, window_size)
            x_1s = torch.from_numpy(f_1s).unsqueeze(0).to(device)
            x_1m = torch.from_numpy(f_1m).unsqueeze(0).to(device)
            x_5m = torch.from_numpy(f_5m).unsqueeze(0).to(device)
            
            _, trend_logits = model(x_1s, x_1m, x_5m)
            probs = torch.softmax(trend_logits, dim=1).cpu().numpy()[0]
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
    model, device = load_model()
    
    print("Loading test data...")
    pf = pq.ParquetFile('data/test.parquet')
    
    day_indices = [0, 50, 100, 200, 500]
    all_pnls, all_trades = [], []
    
    for day_idx in day_indices:
        df = pf.read_row_group(day_idx).to_pandas()
        result = backtest_day(model, device, df)
        print(f"Day {day_idx}: PnL={result['final_pnl']:.2f}, Trades={result['num_trades']}")
        all_pnls.append(result['final_pnl'])
        all_trades.append(result['num_trades'])
    
    print(f"\nSummary over {len(day_indices)} days:")
    print(f"  Total PnL: {sum(all_pnls):.2f}")
    print(f"  Avg PnL/day: {np.mean(all_pnls):.2f}")
    print(f"  Avg trades/day: {np.mean(all_trades):.1f}")


if __name__ == "__main__":
    main()
