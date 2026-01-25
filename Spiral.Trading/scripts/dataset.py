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
    
    def __init__(self, parquet_path: str, window_size: int = 60, stride: int = 1):
        """
        Args:
            parquet_path: Path to parquet file
            window_size: Number of bars per sample
            stride: Step size between windows (1 = every bar, 5 = every 5th bar)
        """
        self.parquet_path = parquet_path
        self.window_size = window_size
        self.stride = stride
        self.pf = pq.ParquetFile(parquet_path)
        
        self.num_row_groups = self.pf.metadata.num_row_groups
        self.bars_per_day = 23400  # 390 minutes * 60 seconds
        self.windows_per_day = (self.bars_per_day - window_size) // stride + 1
        
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
        """Convert global index to (row_group, offset within row group)."""
        row_group = idx // self.windows_per_day
        window_idx = idx % self.windows_per_day
        offset = window_idx * self.stride
        return row_group, offset
    
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
                'open_1m_partial': df['open_1m_partial'].values,
                'high_1m_partial': df['high_1m_partial'].values,
                'low_1m_partial': df['low_1m_partial'].values,
                'close_1m_partial': df['close_1m_partial'].values,
                'open_5m_partial': df['open_5m_partial'].values,
                'high_5m_partial': df['high_5m_partial'].values,
                'low_5m_partial': df['low_5m_partial'].values,
                'close_5m_partial': df['close_5m_partial'].values,
            }
            self._cached_row_group = row_group
    
    def __getitem__(self, idx: int) -> dict:
        row_group, offset = self._get_row_group_and_offset(idx)
        
        # Load row group (cached)
        self._load_row_group(row_group)
        data = self._cached_data
        
        # Current position (end of window)
        end = offset + self.window_size
        pos = end - 1  # 0-indexed position we're predicting at
        
        # 1s features: last 60 seconds
        first_open = data['open'][offset]
        features_1s = np.stack([
            (data['open'][offset:end] - first_open) / first_open,
            (data['high'][offset:end] - first_open) / first_open,
            (data['low'][offset:end] - first_open) / first_open,
            (data['close'][offset:end] - first_open) / first_open,
        ], axis=1).astype(np.float32)  # (60, 4)
        
        # 1m bars: completed bars + current partial
        # Completed 1m bars end at indices 59, 119, 179, ...
        all_1m_indices = np.arange(59, pos + 1, 60)
        if len(all_1m_indices) == 0 or all_1m_indices[-1] != pos:
            all_1m_indices = np.append(all_1m_indices, pos)
        
        # 5m bars: completed bars + current partial  
        # Completed 5m bars end at indices 299, 599, 899, ...
        all_5m_indices = np.arange(299, pos + 1, 300)
        if len(all_5m_indices) == 0 or all_5m_indices[-1] != pos:
            all_5m_indices = np.append(all_5m_indices, pos)
        
        # Build 1m features (last 60 bars)
        max_1m_bars = 60
        features_1m = np.zeros((max_1m_bars, 4), dtype=np.float32)
        n_1m = min(len(all_1m_indices), max_1m_bars)
        indices_1m = all_1m_indices[-n_1m:]
        start_1m = max_1m_bars - n_1m
        features_1m[start_1m:, 0] = (data['open_1m_partial'][indices_1m] - first_open) / first_open
        features_1m[start_1m:, 1] = (data['high_1m_partial'][indices_1m] - first_open) / first_open
        features_1m[start_1m:, 2] = (data['low_1m_partial'][indices_1m] - first_open) / first_open
        features_1m[start_1m:, 3] = (data['close_1m_partial'][indices_1m] - first_open) / first_open
        
        # Build 5m features (max 78 bars per day + 1 partial)
        max_5m_bars = 79
        features_5m = np.zeros((max_5m_bars, 4), dtype=np.float32)
        n_5m = min(len(all_5m_indices), max_5m_bars)
        indices_5m = all_5m_indices[-n_5m:]
        start_5m = max_5m_bars - n_5m
        features_5m[start_5m:, 0] = (data['open_5m_partial'][indices_5m] - first_open) / first_open
        features_5m[start_5m:, 1] = (data['high_5m_partial'][indices_5m] - first_open) / first_open
        features_5m[start_5m:, 2] = (data['low_5m_partial'][indices_5m] - first_open) / first_open
        features_5m[start_5m:, 3] = (data['close_5m_partial'][indices_5m] - first_open) / first_open
        
        # Labels: use the label at the end of the window
        session = data['session'][pos]
        trend = data['trend'][pos]
        
        return {
            'features_1s': torch.from_numpy(features_1s),   # (60, 4)
            'features_1m': torch.from_numpy(features_1m),   # (391, 4)
            'features_5m': torch.from_numpy(features_5m),   # (79, 4)
            'session': torch.tensor(session, dtype=torch.long),
            'trend': torch.tensor(trend, dtype=torch.long),
        }


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
