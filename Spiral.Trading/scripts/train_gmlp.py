#!/usr/bin/env python3
"""gMLP - Gated MLP with Spatial Gating Unit for trading classification."""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TradingDataset, RowGroupSampler


class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit: s(Z) = Z1 * f(Z2) where f(Z2) = W @ Z2 + b"""
    def __init__(self, d_ffn: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.weight = nn.Parameter(torch.zeros(seq_len, seq_len).uniform_(-0.01, 0.01))
        self.bias = nn.Parameter(torch.ones(seq_len))
    
    def forward(self, z):
        # z: [batch, seq_len, d_ffn]
        z1, z2 = z.chunk(2, dim=-1)
        z2 = self.norm(z2)
        # Spatial projection: [seq, seq] @ [batch, seq, d] -> [batch, seq, d]
        z2 = torch.einsum('ij,bjd->bid', self.weight, z2) + self.bias.unsqueeze(-1)
        return z1 * z2


class GMLPBlock(nn.Module):
    """gMLP Block: Z = GELU(X @ U), Z_tilde = SGU(Z), Y = Z_tilde @ V"""
    def __init__(self, d_model: int, d_ffn: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.proj2 = nn.Linear(d_ffn // 2, d_model)
    
    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        z = self.activation(self.proj1(x))
        z = self.sgu(z)
        return self.proj2(z) + shortcut


class TradingGMLP(nn.Module):
    """gMLP for trading classification with pre-normalized features."""
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 128,
        ffn_dim: int = 512,
        num_layers: int = 4,
        num_sessions: int = 3,
        num_trends: int = 7,
    ):
        super().__init__()
        total_patches = 60 + 60 + 78  # 198
        
        self.embed = nn.Linear(input_channels, hidden_dim)
        
        self.blocks = nn.Sequential(*[
            GMLPBlock(hidden_dim, ffn_dim, total_patches)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.session_head = nn.Linear(hidden_dim, num_sessions)
        self.trend_head = nn.Linear(hidden_dim, num_trends)
    
    def forward(self, x_1s, x_1m, x_5m):
        x = torch.cat([x_1s, x_1m, x_5m], dim=1)
        x = self.embed(x)
        x = self.blocks(x)
        x = self.norm(x.mean(dim=1))
        return self.session_head(x), self.trend_head(x)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    
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
            elapsed = time.time() - start_time
            samples_per_sec = total / elapsed if elapsed > 0 else 0
            print(f"  Batch {batch_idx}: loss={loss.item():.4f}, {samples_per_sec:.0f} samples/sec")
    
    elapsed = time.time() - start_time
    return {'loss': total_loss / len(loader), 'acc': correct / total, 'time': elapsed}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    
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
    
    elapsed = time.time() - start_time
    return {'loss': total_loss / len(loader), 'acc': correct / total, 'time': elapsed}


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
    
    model = TradingGMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    total_start = time.time()
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, Time: {train_metrics['time']:.1f}s")
        
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['acc']:.4f}, Time: {test_metrics['time']:.1f}s")
    
    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")
    torch.save(model.state_dict(), 'data/gmlp_model.pt')
    print("Saved model to data/gmlp_model.pt")


if __name__ == '__main__':
    main()
