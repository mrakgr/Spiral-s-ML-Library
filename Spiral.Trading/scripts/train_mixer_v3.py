#!/usr/bin/env python3
"""MLP-Mixer v3 - bar-relative normalization for better pattern matching."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import TradingDataset, RowGroupSampler

# Import the v3 dataset from transformer_v3
from train_transformer_v3 import TradingDatasetV3


class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, num_patches, channels, token_hidden, channel_hidden):
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.token_mix = MLPBlock(num_patches, token_hidden)
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_mix = MLPBlock(channels, channel_hidden)
    
    def forward(self, x):
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mix(y)
        y = y.transpose(1, 2)
        x = x + y
        
        y = self.channel_norm(x)
        y = self.channel_mix(y)
        x = x + y
        return x


class TradingMixerV3(nn.Module):
    """Unified MLP-Mixer with bar-relative features (3 channels)."""
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_sessions: int = 3,
        num_trends: int = 7,
    ):
        super().__init__()
        total_patches = 60 + 60 + 78  # 198
        
        self.embed = nn.Linear(input_channels, hidden_dim)
        
        self.mixer = nn.Sequential(*[
            MixerBlock(total_patches, hidden_dim, hidden_dim * 2, hidden_dim * 2)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.session_head = nn.Linear(hidden_dim, num_sessions)
        self.trend_head = nn.Linear(hidden_dim, num_trends)
    
    def forward(self, x_1s, x_1m, x_5m):
        x = torch.cat([x_1s, x_1m, x_5m], dim=1)
        x = self.embed(x)
        x = self.mixer(x)
        x = self.norm(x.mean(dim=1))
        return self.session_head(x), self.trend_head(x)


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
    
    model = TradingMixerV3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['acc']:.4f}")
    
    torch.save(model.state_dict(), 'data/mixer_v3_model.pt')
    print("\nSaved model to data/mixer_v3_model.pt")


if __name__ == '__main__':
    main()
