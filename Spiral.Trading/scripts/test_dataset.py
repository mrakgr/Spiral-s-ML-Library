#!/usr/bin/env python3
"""Test the TradingDataset class."""

from dataset import TradingDataset
from torch.utils.data import DataLoader


def main():
    print("Loading dataset...")
    dataset = TradingDataset('data/train.parquet', window_size=60)
    
    print(f"Dataset size: {len(dataset):,} samples")
    print(f"Row groups: {dataset.num_row_groups}")
    print(f"Windows per day: {dataset.windows_per_day}")
    print()
    
    print("=== First sample ===")
    sample = dataset[0]
    print(f"Features shape: {sample['features'].shape}")
    print(f"Session: {sample['session'].item()}")
    print(f"Trend: {sample['trend'].item()}")
    print(f"Features (first 5 bars):\n{sample['features'][:5]}")
    print()
    
    print("=== Testing DataLoader ===")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch features shape: {batch['features'].shape}")
    print(f"Batch session shape: {batch['session'].shape}")
    print(f"Batch trend shape: {batch['trend'].shape}")


if __name__ == "__main__":
    main()
