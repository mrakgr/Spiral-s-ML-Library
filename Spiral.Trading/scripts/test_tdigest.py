#!/usr/bin/env python3
"""Test t-digest normalization on 1s bar data."""

import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from pytdigest import TDigest


def load_day_data(path, row_group=0):
    """Load one day of 1s bar data."""
    pf = pq.ParquetFile(path)
    table = pf.read_row_group(row_group)
    return {
        'open': table['open'].to_numpy(),
        'high': table['high'].to_numpy(),
        'low': table['low'].to_numpy(),
        'close': table['close'].to_numpy(),
    }


def compute_deltas(data):
    """Compute price deltas: close - previous close."""
    close = data['close']
    deltas = np.diff(close)
    return deltas


def tdigest_transform(values, td):
    """Transform values via t-digest CDF, rescaled to [-1, 1]."""
    # CDF gives [0, 1], rescale to [-1, 1]
    cdf_values = np.array([td.cdf(v) for v in values])
    return 2 * cdf_values - 1


def plot_comparison(raw, transformed, title_suffix=""):
    """Plot histograms comparing raw vs transformed distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Raw histogram
    ax1 = axes[0, 0]
    ax1.hist(raw, bins=100, edgecolor='black', alpha=0.7)
    ax1.set_title(f'Raw Deltas{title_suffix}')
    ax1.set_xlabel('Delta')
    ax1.set_ylabel('Count')
    
    # Raw histogram (log scale)
    ax2 = axes[0, 1]
    ax2.hist(raw, bins=100, edgecolor='black', alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_title(f'Raw Deltas (log scale){title_suffix}')
    ax2.set_xlabel('Delta')
    ax2.set_ylabel('Count (log)')
    
    # Transformed histogram
    ax3 = axes[1, 0]
    ax3.hist(transformed, bins=100, edgecolor='black', alpha=0.7)
    ax3.set_title(f'T-Digest Transformed [-1, 1]{title_suffix}')
    ax3.set_xlabel('Transformed Value')
    ax3.set_ylabel('Count')
    
    # Transformed histogram (should be roughly uniform)
    ax4 = axes[1, 1]
    ax4.hist(transformed, bins=100, edgecolor='black', alpha=0.7)
    ax4.set_yscale('log')
    ax4.set_title(f'T-Digest Transformed (log scale){title_suffix}')
    ax4.set_xlabel('Transformed Value')
    ax4.set_ylabel('Count (log)')
    
    plt.tight_layout()
    return fig


def main():
    print("Loading day 0 data...")
    data = load_day_data('data/train.parquet', row_group=0)
    print(f"Loaded {len(data['close']):,} bars")
    
    print("\nComputing price deltas (close - prev close)...")
    deltas = compute_deltas(data)
    print(f"Deltas: {len(deltas):,} values")
    print(f"  Min: {deltas.min():.6f}")
    print(f"  Max: {deltas.max():.6f}")
    print(f"  Mean: {deltas.mean():.6f}")
    print(f"  Std: {deltas.std():.6f}")
    
    print("\nBuilding t-digest from full day...")
    td = TDigest.compute(deltas, compression=1000)
    print(f"T-digest centroids: {len(td.get_centroids())}")
    
    print("\nTransforming via t-digest CDF...")
    transformed = tdigest_transform(deltas, td)
    print(f"Transformed range: [{transformed.min():.4f}, {transformed.max():.4f}]")
    print(f"Transformed mean: {transformed.mean():.4f}")
    print(f"Transformed std: {transformed.std():.4f}")
    
    # Check quantiles
    print("\nQuantile check:")
    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        raw_q = np.quantile(deltas, q)
        trans_q = np.quantile(transformed, q)
        print(f"  {q:.2f}: raw={raw_q:+.6f}, transformed={trans_q:+.4f}")
    
    print("\nGenerating comparison plots...")
    fig = plot_comparison(deltas, transformed)
    fig.savefig('data/tdigest_comparison.png', dpi=150)
    print("Saved to data/tdigest_comparison.png")
    plt.close()


if __name__ == '__main__':
    main()
