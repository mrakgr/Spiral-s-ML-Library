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
    
    def __init__(self, parquet_path: str, window_size: int = 60):
        """
        Args:
            parquet_path: Path to parquet file
            window_size: Number of bars per sample
        """
        self.parquet_path = parquet_path
        self.window_size = window_size
        self.pf = pq.ParquetFile(parquet_path)
        
        self.num_row_groups = self.pf.metadata.num_row_groups
        self.bars_per_day = 23400  # 390 minutes * 60 seconds
        self.windows_per_day = self.bars_per_day - window_size + 1
        
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
        offset = idx % self.windows_per_day
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
            }
            self._cached_row_group = row_group
    
    def __getitem__(self, idx: int) -> dict:
        row_group, offset = self._get_row_group_and_offset(idx)
        
        # Load row group (cached)
        self._load_row_group(row_group)
        data = self._cached_data
        
        # Extract window
        end = offset + self.window_size
        first_open = data['open'][offset]
        
        features = np.stack([
            (data['open'][offset:end] - first_open) / first_open,
            (data['high'][offset:end] - first_open) / first_open,
            (data['low'][offset:end] - first_open) / first_open,
            (data['close'][offset:end] - first_open) / first_open,
        ], axis=1).astype(np.float32)
        
        # Labels: use the label at the end of the window
        session = data['session'][end - 1]
        trend = data['trend'][end - 1]
        
        return {
            'features': torch.from_numpy(features),
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
