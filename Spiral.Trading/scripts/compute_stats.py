#!/usr/bin/env python3
"""Compute input statistics for normalization."""

import numpy as np
from train_transformer_v3 import TradingDatasetV3

def main():
    print("Loading dataset...")
    ds = TradingDatasetV3('data/train.parquet', window_size=60, stride=5)
    
    n_samples = 100_000
    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)
    
    # Collect samples for each timeframe
    samples_1s = []
    samples_1m = []
    samples_5m = []
    
    print(f"Sampling {n_samples:,} examples...")
    for i, idx in enumerate(indices):
        sample = ds[idx]
        samples_1s.append(sample['features_1s'].numpy())
        samples_1m.append(sample['features_1m'].numpy())
        samples_5m.append(sample['features_5m'].numpy())
        
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1:,}/{n_samples:,}")
    
    # Stack into arrays
    arr_1s = np.stack(samples_1s)  # (n_samples, 60, 3)
    arr_1m = np.stack(samples_1m)  # (n_samples, 60, 3)
    arr_5m = np.stack(samples_5m)  # (n_samples, 78, 3)
    
    print("\n=== 1s features ===")
    for c, name in enumerate(['H-O', 'L-O', 'C-O']):
        vals = arr_1s[:, :, c].flatten()
        print(f"  {name}: mean={vals.mean():.6f}, std={vals.std():.6f}")
    
    print("\n=== 1m features ===")
    for c, name in enumerate(['H-O', 'L-O', 'C-O']):
        vals = arr_1m[:, :, c].flatten()
        print(f"  {name}: mean={vals.mean():.6f}, std={vals.std():.6f}")
    
    print("\n=== 5m features ===")
    for c, name in enumerate(['H-O', 'L-O', 'C-O']):
        vals = arr_5m[:, :, c].flatten()
        print(f"  {name}: mean={vals.mean():.6f}, std={vals.std():.6f}")


if __name__ == '__main__':
    main()
