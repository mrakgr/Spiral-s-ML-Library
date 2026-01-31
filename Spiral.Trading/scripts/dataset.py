"""PyTorch Dataset for loading trading simulation data from Parquet."""

import torch
from torch.utils.data import Dataset, Sampler
import pyarrow.parquet as pq
import numpy as np


class TradingDataset(Dataset):
    """
    Dataset that loads trading bars from parquet file.
    
    Each sample is a window of consecutive bars.
    Features: OHLC (normalized as returns)
    Labels: session (0-2) and trend (0-6)
    """
    
    def __init__(self, parquet_path: str, window_size: int = 60, stride: int = 1, start_index: int = 0):
        """
        Args:
            parquet_path: Path to parquet file
            window_size: Number of bars per sample
            stride: Step size between windows (1 = every bar, 5 = every 5th bar)
            start_index: First position to sample from (0 = start of day with padding)
        """
        self.parquet_path = parquet_path
        self.window_size = window_size
        self.stride = stride
        self.start_index = start_index
        self.pf = pq.ParquetFile(parquet_path)
        
        self.num_row_groups = self.pf.metadata.num_row_groups
        self.bars_per_day = 23400  # 390 minutes * 60 seconds
        self.windows_per_day = (self.bars_per_day - start_index) // stride
        
        # Verify row group size matches expected bars per day
        expected_rows = self.num_row_groups * self.bars_per_day
        actual_rows = self.pf.metadata.num_rows
        assert actual_rows == expected_rows, \
            f"Expected {expected_rows} rows ({self.num_row_groups} days × {self.bars_per_day} bars), got {actual_rows}"
        
        # Cache for row group data
        self._cached_row_group = -1
        self._cached_data = None
        
    def __len__(self) -> int:
        return self.num_row_groups * self.windows_per_day
    
    def _get_row_group_and_offset(self, idx: int) -> tuple[int, int]:
        """Convert global index to (row_group, position within row group)."""
        row_group = idx // self.windows_per_day
        window_idx = idx % self.windows_per_day
        pos = self.start_index + window_idx * self.stride
        return row_group, pos
    
    def _load_row_group(self, row_group: int):
        """Load and cache a row group as numpy arrays."""
        if self._cached_row_group != row_group:
            df = self.pf.read_row_group(row_group).to_pandas()
            self._cached_data = {
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'session': df['session'].values,
                'trend': df['trend'].values,
                'cdf_high_1s': df['cdf_high_1s'].values,
                'cdf_low_1s': df['cdf_low_1s'].values,
                'cdf_close_1s': df['cdf_close_1s'].values,
                'cdf_high_1m': df['cdf_high_1m'].values,
                'cdf_low_1m': df['cdf_low_1m'].values,
                'cdf_close_1m': df['cdf_close_1m'].values,
                'cdf_high_5m': df['cdf_high_5m'].values,
                'cdf_low_5m': df['cdf_low_5m'].values,
                'cdf_close_5m': df['cdf_close_5m'].values,
            }
            self._cached_row_group = row_group

    def get_features_for(self,row_group : int, pos : int) -> dict:
        assert 0 <= pos < self.bars_per_day, f"pos {pos} out of range [0, {self.bars_per_day})"
        self._load_row_group(row_group)
        data = self._cached_data
        
        # Build 1s features with padding if needed (already CDF-normalized)
        features_1s = np.zeros((self.window_size, 3), dtype=np.float32)
        available = min(pos + 1, self.window_size)
        start_idx = max(0, pos + 1 - self.window_size)
        start_pad = self.window_size - available
        
        features_1s[start_pad:, 0] = data['cdf_high_1s'][start_idx:pos + 1]
        features_1s[start_pad:, 1] = data['cdf_low_1s'][start_idx:pos + 1]
        features_1s[start_pad:, 2] = data['cdf_close_1s'][start_idx:pos + 1]
        
        # 1m bars: completed bars + current partial
        all_1m_indices = np.arange(59, pos + 1, 60)
        if len(all_1m_indices) == 0 or all_1m_indices[-1] != pos:
            all_1m_indices = np.append(all_1m_indices, pos)
        
        # 5m bars: completed bars + current partial  
        all_5m_indices = np.arange(299, pos + 1, 300)
        if len(all_5m_indices) == 0 or all_5m_indices[-1] != pos:
            all_5m_indices = np.append(all_5m_indices, pos)
        
        # Build 1m features (already CDF-normalized, last 60 bars)
        max_1m_bars = 60
        features_1m = np.zeros((max_1m_bars, 3), dtype=np.float32)
        n_1m = min(len(all_1m_indices), max_1m_bars)
        indices_1m = all_1m_indices[-n_1m:]
        start_1m = max_1m_bars - n_1m
        features_1m[start_1m:, 0] = data['cdf_high_1m'][indices_1m]
        features_1m[start_1m:, 1] = data['cdf_low_1m'][indices_1m]
        features_1m[start_1m:, 2] = data['cdf_close_1m'][indices_1m]
        
        # Build 5m features (already CDF-normalized, max 78 bars per day)
        max_5m_bars = 78
        features_5m = np.zeros((max_5m_bars, 3), dtype=np.float32)
        n_5m = min(len(all_5m_indices), max_5m_bars)
        indices_5m = all_5m_indices[-n_5m:]
        start_5m = max_5m_bars - n_5m
        features_5m[start_5m:, 0] = data['cdf_high_5m'][indices_5m]
        features_5m[start_5m:, 1] = data['cdf_low_5m'][indices_5m]
        features_5m[start_5m:, 2] = data['cdf_close_5m'][indices_5m]
        
        # Labels: use the label at the end of the window
        session = data['session'][pos]
        trend = data['trend'][pos]
        
        return {
            'features_1s': torch.from_numpy(features_1s),   # (60, 3)
            'features_1m': torch.from_numpy(features_1m),   # (60, 3)
            'features_5m': torch.from_numpy(features_5m),   # (78, 3)
            'session': torch.tensor(session, dtype=torch.long),
            'trend': torch.tensor(trend, dtype=torch.long),
        }
    
    def __getitem__(self, idx: int) -> dict:
        row_group, pos = self._get_row_group_and_offset(idx)
        return self.get_features_for(row_group,pos)

class RowGroupSampler(Sampler):
    """
    Sampler that shuffles row groups but iterates sequentially within each.
    This makes caching effective while still providing randomization.
    """
    
    def __init__(self, dataset: TradingDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_row_groups = dataset.num_row_groups
        self.windows_per_day = dataset.windows_per_day
    
    def __iter__(self):
        row_groups = list(range(self.num_row_groups))
        if self.shuffle:
            np.random.shuffle(row_groups)
        
        for rg in row_groups:
            base = rg * self.windows_per_day
            for offset in range(self.windows_per_day):
                yield base + offset
    
    def __len__(self):
        return len(self.dataset)
