#!/usr/bin/env python3
"""Transformer v3 - bar-relative normalization for better pattern matching."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import TradingDataset, RowGroupSampler
from x_transformers import ContinuousTransformerWrapper, Encoder
import pyarrow.parquet as pq


class TradingDatasetV3(TradingDataset):
    """Dataset with bar-relative normalization (each bar normalized by its own open)."""
    
    def __getitem__(self, idx: int) -> dict:
        row_group, offset = self._get_row_group_and_offset(idx)
        self._load_row_group(row_group)
        data = self._cached_data
        
        end = offset + self.window_size
        pos = end - 1
        
        # 1s features: normalize each bar by its own open
        opens_1s = data['open'][offset:end]
        features_1s = np.stack([
            (data['high'][offset:end] - opens_1s) / opens_1s,
            (data['low'][offset:end] - opens_1s) / opens_1s,
            (data['close'][offset:end] - opens_1s) / opens_1s,
        ], axis=1).astype(np.float32)  # (60, 3)
        
        # 1m indices
        all_1m_indices = np.arange(59, pos + 1, 60)
        if len(all_1m_indices) == 0 or all_1m_indices[-1] != pos:
            all_1m_indices = np.append(all_1m_indices, pos)
        
        # 5m indices
        all_5m_indices = np.arange(299, pos + 1, 300)
        if len(all_5m_indices) == 0 or all_5m_indices[-1] != pos:
            all_5m_indices = np.append(all_5m_indices, pos)
        
        # Build 1m features (bar-relative)
        max_1m_bars = 60
        features_1m = np.zeros((max_1m_bars, 3), dtype=np.float32)
        n_1m = min(len(all_1m_indices), max_1m_bars)
        indices_1m = all_1m_indices[-n_1m:]
        start_1m = max_1m_bars - n_1m
        opens_1m = data['open_1m_partial'][indices_1m]
        features_1m[start_1m:, 0] = (data['high_1m_partial'][indices_1m] - opens_1m) / opens_1m
        features_1m[start_1m:, 1] = (data['low_1m_partial'][indices_1m] - opens_1m) / opens_1m
        features_1m[start_1m:, 2] = (data['close_1m_partial'][indices_1m] - opens_1m) / opens_1m
        
        # Build 5m features (bar-relative)
        max_5m_bars = 78
        features_5m = np.zeros((max_5m_bars, 3), dtype=np.float32)
        n_5m = min(len(all_5m_indices), max_5m_bars)
        indices_5m = all_5m_indices[-n_5m:]
        start_5m = max_5m_bars - n_5m
        opens_5m = data['open_5m_partial'][indices_5m]
        features_5m[start_5m:, 0] = (data['high_5m_partial'][indices_5m] - opens_5m) / opens_5m
        features_5m[start_5m:, 1] = (data['low_5m_partial'][indices_5m] - opens_5m) / opens_5m
        features_5m[start_5m:, 2] = (data['close_5m_partial'][indices_5m] - opens_5m) / opens_5m
        
        session = data['session'][pos]
        trend = data['trend'][pos]
        
        return {
            'features_1s': torch.from_numpy(features_1s),
            'features_1m': torch.from_numpy(features_1m),
            'features_5m': torch.from_numpy(features_5m),
            'session': torch.tensor(session, dtype=torch.long),
            'trend': torch.tensor(trend, dtype=torch.long),
        }


class TradingTransformerV3(nn.Module):
    """Unified Transformer with bar-relative features (3 channels: H, L, C relative to O)."""
    def __init__(
        self,
        input_channels: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_sessions: int = 3,
        num_trends: int = 7,
    ):
        super().__init__()
        
        self.transformer = ContinuousTransformerWrapper(
            dim_in=input_channels,
            dim_out=d_model,
            max_seq_len=198,
            attn_layers=Encoder(
                dim=d_model,
                depth=num_layers,
                heads=nhead,
                rotary_pos_emb=True
            )
        )
        
        self.session_head = nn.Linear(d_model, num_sessions)
        self.trend_head = nn.Linear(d_model, num_trends)
    
    def forward(self, x_1s, x_1m, x_5m):
        x = torch.cat([x_1s, x_1m, x_5m], dim=1)  # (batch, 198, 3)
        h = self.transformer(x)[:, -1, :]  # (batch, d_model)
        return self.session_head(h), self.trend_head(h)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(loader):
        x_1s = batch['features_1s'].to(device)
        x_1m = batch['features_1m'].to(device)
        x_5m = batch['features_5m'].to(device)
        labels = batch['trend'].to(device)
        
        optimizer.zero_grad()
        _, trend_logits = model(x_1s, x_1m, x_5m)
        loss = criterion(trend_logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (trend_logits.argmax(1) == labels).sum().item()
        total += x_1s.size(0)
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: loss={loss.item():.4f}")
    
    return {'loss': total_loss / len(loader), 'acc': correct / total}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    
    for batch in loader:
        x_1s = batch['features_1s'].to(device)
        x_1m = batch['features_1m'].to(device)
        x_5m = batch['features_5m'].to(device)
        labels = batch['trend'].to(device)
        
        _, trend_logits = model(x_1s, x_1m, x_5m)
        loss = criterion(trend_logits, labels)
        
        total_loss += loss.item()
        correct += (trend_logits.argmax(1) == labels).sum().item()
        total += x_1s.size(0)
    
    return {'loss': total_loss / len(loader), 'acc': correct / total}


def main():
    batch_size = 256
    num_epochs = 1
    lr = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_ds = TradingDatasetV3('data/train.parquet', window_size=60, stride=5)
    test_ds = TradingDatasetV3('data/test.parquet', window_size=60, stride=5)
    print(f"Train: {len(train_ds):,}, Test: {len(test_ds):,}")
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=RowGroupSampler(train_ds, shuffle=True), num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        sampler=RowGroupSampler(test_ds, shuffle=False), num_workers=0
    )
    
    model = TradingTransformerV3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['acc']:.4f}")
    
    torch.save(model.state_dict(), 'data/transformer_v3_model.pt')
    print("\nSaved model to data/transformer_v3_model.pt")


if __name__ == '__main__':
    main()
