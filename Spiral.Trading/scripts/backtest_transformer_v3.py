#!/usr/bin/env python3
"""Backtest for Transformer v3 with bar-relative normalization."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_transformer_v3 import TradingTransformerV3

TREND_NAMES = [
    'StrongUptrend', 'MidUptrend', 'WeakUptrend', 'Consolidation',
    'WeakDowntrend', 'MidDowntrend', 'StrongDowntrend'
]

TREND_POSITION = {0: 1, 1: 1, 2: 1, 3: 0, 4: -1, 5: -1, 6: -1}


def load_model(path='data/transformer_v3_model.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingTransformerV3().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, device


def build_features_v3(df, i, window_size=60):
    """Build bar-relative multi-timeframe features."""
    end = i + window_size
    pos = end - 1
    
    # 1s features (bar-relative)
    opens_1s = df['open'].iloc[i:end].values
    f_1s = np.stack([
        (df['high'].iloc[i:end].values - opens_1s) / opens_1s,
        (df['low'].iloc[i:end].values - opens_1s) / opens_1s,
        (df['close'].iloc[i:end].values - opens_1s) / opens_1s,
    ], axis=1).astype(np.float32)
    
    # 1m indices
    all_1m = np.arange(59, pos + 1, 60)
    if len(all_1m) == 0 or all_1m[-1] != pos:
        all_1m = np.append(all_1m, pos)
    
    # 5m indices
    all_5m = np.arange(299, pos + 1, 300)
    if len(all_5m) == 0 or all_5m[-1] != pos:
        all_5m = np.append(all_5m, pos)
    
    # 1m features (bar-relative)
    f_1m = np.zeros((60, 3), dtype=np.float32)
    n_1m = min(len(all_1m), 60)
    idx_1m = all_1m[-n_1m:]
    start_1m = 60 - n_1m
    opens_1m = df['open_1m_partial'].iloc[idx_1m].values
    f_1m[start_1m:, 0] = (df['high_1m_partial'].iloc[idx_1m].values - opens_1m) / opens_1m
    f_1m[start_1m:, 1] = (df['low_1m_partial'].iloc[idx_1m].values - opens_1m) / opens_1m
    f_1m[start_1m:, 2] = (df['close_1m_partial'].iloc[idx_1m].values - opens_1m) / opens_1m
    
    # 5m features (bar-relative)
    f_5m = np.zeros((78, 3), dtype=np.float32)
    n_5m = min(len(all_5m), 78)
    idx_5m = all_5m[-n_5m:]
    start_5m = 78 - n_5m
    opens_5m = df['open_5m_partial'].iloc[idx_5m].values
    f_5m[start_5m:, 0] = (df['high_5m_partial'].iloc[idx_5m].values - opens_5m) / opens_5m
    f_5m[start_5m:, 1] = (df['low_5m_partial'].iloc[idx_5m].values - opens_5m) / opens_5m
    f_5m[start_5m:, 2] = (df['close_5m_partial'].iloc[idx_5m].values - opens_5m) / opens_5m
    
    return f_1s, f_1m, f_5m


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


def backtest_day(model, device, df, window_size=60, stride=5):
    system = TradingSystem(entry_threshold=0.90)
    prices = df['close'].values
    positions, pnls, times, pred_trends = [], [], [], []
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
            pred_trends.append(probs.argmax())
            times.append(i + window_size)
            
            if i + window_size + stride < len(prices):
                price_change = prices[i + window_size + stride] - prices[i + window_size]
                cumulative_pnl += position * price_change
            pnls.append(cumulative_pnl)
    
    return {
        'times': np.array(times), 'positions': np.array(positions),
        'pnls': np.array(pnls), 'final_pnl': cumulative_pnl,
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
