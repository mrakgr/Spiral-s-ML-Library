#!/usr/bin/env python3
"""LightGBM baseline for trend classification."""

import numpy as np
from dataset import TradingDataset
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import time


def load_features(path: str, max_rows: int = None):
    """Load data using TradingDataset and flatten features."""
    print(f"Loading {path}...")
    ds = TradingDataset(path, window_size=60, stride=5)
    
    n = min(len(ds), max_rows) if max_rows else len(ds)
    
    # Sample evenly across dataset
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)
    
    all_features = []
    all_labels = []
    
    for i, idx in enumerate(indices):
        sample = ds[idx]
        
        # Flatten 1s features
        f_1s = sample['features_1s'].numpy().flatten()  # (240,)
        
        # Summary stats for 1m (mean, std, min, max per channel)
        f_1m = sample['features_1m'].numpy()  # (60, 4)
        f_1m_stats = np.array([
            f_1m[:, 0].mean(), f_1m[:, 0].std(), f_1m[:, 0].min(), f_1m[:, 0].max(),
            f_1m[:, 1].mean(), f_1m[:, 1].std(), f_1m[:, 1].min(), f_1m[:, 1].max(),
            f_1m[:, 2].mean(), f_1m[:, 2].std(), f_1m[:, 2].min(), f_1m[:, 2].max(),
            f_1m[:, 3].mean(), f_1m[:, 3].std(), f_1m[:, 3].min(), f_1m[:, 3].max(),
        ])  # (16,)
        
        # Summary stats for 5m
        f_5m = sample['features_5m'].numpy()  # (78, 4)
        f_5m_stats = np.array([
            f_5m[:, 0].mean(), f_5m[:, 0].std(), f_5m[:, 0].min(), f_5m[:, 0].max(),
            f_5m[:, 1].mean(), f_5m[:, 1].std(), f_5m[:, 1].min(), f_5m[:, 1].max(),
            f_5m[:, 2].mean(), f_5m[:, 2].std(), f_5m[:, 2].min(), f_5m[:, 2].max(),
            f_5m[:, 3].mean(), f_5m[:, 3].std(), f_5m[:, 3].min(), f_5m[:, 3].max(),
        ])  # (16,)
        
        features = np.concatenate([f_1s, f_1m_stats, f_5m_stats])  # (272,)
        all_features.append(features)
        all_labels.append(sample['trend'].item())
        
        if (i + 1) % 100000 == 0:
            print(f"  Loaded {i + 1:,}/{n:,}")
    
    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def main():
    start = time.time()
    
    # Load data (use subset for speed)
    X_train, y_train = load_features('data/train.parquet', max_rows=1_000_000)
    X_test, y_test = load_features('data/test.parquet', max_rows=100_000)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Data loading: {time.time() - start:.1f}s")
    
    # Replace NaN with 0 (from std of single element)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Train LightGBM
    train_start = time.time()
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
        'objective': 'multiclass',
        'num_class': 7,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'verbose': -1,
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(10)],
    )
    
    print(f"Training: {time.time() - train_start:.1f}s")
    
    # Evaluate
    y_pred_train = model.predict(X_train).argmax(axis=1)
    y_pred_test = model.predict(X_test).argmax(axis=1)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == '__main__':
    main()
