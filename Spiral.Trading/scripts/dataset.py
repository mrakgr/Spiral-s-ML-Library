"""PyTorch Dataset for loading trading simulation data from Parquet."""

import torch
from torch.utils.data import Dataset
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
        
    def __len__(self) -> int:
        return self.num_row_groups * self.windows_per_day
    
    def _get_row_group_and_offset(self, idx: int) -> tuple[int, int]:
        """Convert global index to (row_group, offset within row group)."""
        row_group = idx // self.windows_per_day
        offset = idx % self.windows_per_day
        return row_group, offset
    
    def __getitem__(self, idx: int) -> dict:
        row_group, offset = self._get_row_group_and_offset(idx)
        
        # Load the row group (cached by PyArrow)
        df = self.pf.read_row_group(row_group).to_pandas()
        
        # Extract window
        window = df.iloc[offset:offset + self.window_size]
        
        # Features: OHLC as returns (percentage change from first open)
        first_open = window['open'].iloc[0]
        features = np.stack([
            (window['open'].values - first_open) / first_open,
            (window['high'].values - first_open) / first_open,
            (window['low'].values - first_open) / first_open,
            (window['close'].values - first_open) / first_open,
        ], axis=1).astype(np.float32)
        
        # Labels: use the label at the end of the window
        session = window['session'].iloc[-1]
        trend = window['trend'].iloc[-1]
        
        return {
            'features': torch.from_numpy(features),  # (window_size, 4)
            'session': torch.tensor(session, dtype=torch.long),
            'trend': torch.tensor(trend, dtype=torch.long),
        }
