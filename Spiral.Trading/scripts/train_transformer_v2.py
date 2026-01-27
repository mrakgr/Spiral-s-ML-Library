#!/usr/bin/env python3
"""Transformer v2 - unified cross-timeframe processing with x-transformers and rotary embeddings."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import TradingDataset, RowGroupSampler
from x_transformers import ContinuousTransformerWrapper, Encoder


class TradingTransformerV2(nn.Module):
    """
    Unified Transformer - concatenates all timeframes and processes together.
    Total patches: 60 (1s) + 60 (1m) + 78 (5m) = 198
    Uses x-transformers with rotary position embeddings.
    """
    def __init__(
        self,
        input_channels: int = 4,
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
        x = torch.cat([x_1s, x_1m, x_5m], dim=1)  # (batch, 198, 4)
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
    train_ds = TradingDataset('data/train.parquet', window_size=60, stride=5)
    test_ds = TradingDataset('data/test.parquet', window_size=60, stride=5)
    print(f"Train: {len(train_ds):,}, Test: {len(test_ds):,}")
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=RowGroupSampler(train_ds, shuffle=True), num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        sampler=RowGroupSampler(test_ds, shuffle=False), num_workers=0
    )
    
    model = TradingTransformerV2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['acc']:.4f}")
    
    torch.save(model.state_dict(), 'data/transformer_v2_model.pt')
    print("\nSaved model to data/transformer_v2_model.pt")


if __name__ == '__main__':
    main()
