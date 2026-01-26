#!/usr/bin/env python3
"""MLP-Mixer model for trend classification."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import TradingDataset, RowGroupSampler


class MLPBlock(nn.Module):
    """Simple MLP block with GELU activation."""
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
    """
    Mixer block with token mixing and channel mixing.
    Input: (batch, num_patches, channels)
    """
    def __init__(self, num_patches, channels, token_hidden, channel_hidden):
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.token_mix = MLPBlock(num_patches, token_hidden)
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_mix = MLPBlock(channels, channel_hidden)
    
    def forward(self, x):
        # Token mixing: mix across patches (transpose to mix along patch dim)
        y = self.token_norm(x)
        y = y.transpose(1, 2)  # (batch, channels, num_patches)
        y = self.token_mix(y)
        y = y.transpose(1, 2)  # (batch, num_patches, channels)
        x = x + y
        
        # Channel mixing: mix across channels
        y = self.channel_norm(x)
        y = self.channel_mix(y)
        x = x + y
        
        return x


class TradingMixer(nn.Module):
    """
    Multi-timeframe MLP-Mixer for trend classification.
    
    Each timeframe gets its own mixer, outputs are concatenated.
    """
    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_sessions: int = 3,
        num_trends: int = 7,
    ):
        super().__init__()
        
        # Patch embedding (project input channels to hidden dim)
        self.embed_1s = nn.Linear(input_channels, hidden_dim)
        self.embed_1m = nn.Linear(input_channels, hidden_dim)
        self.embed_5m = nn.Linear(input_channels, hidden_dim)
        
        # Mixer blocks for each timeframe
        self.mixer_1s = nn.Sequential(*[
            MixerBlock(60, hidden_dim, hidden_dim * 2, hidden_dim * 2)
            for _ in range(num_layers)
        ])
        self.mixer_1m = nn.Sequential(*[
            MixerBlock(60, hidden_dim, hidden_dim * 2, hidden_dim * 2)
            for _ in range(num_layers)
        ])
        self.mixer_5m = nn.Sequential(*[
            MixerBlock(78, hidden_dim, hidden_dim * 2, hidden_dim * 2)
            for _ in range(num_layers)
        ])
        
        # Final layer norms
        self.norm_1s = nn.LayerNorm(hidden_dim)
        self.norm_1m = nn.LayerNorm(hidden_dim)
        self.norm_5m = nn.LayerNorm(hidden_dim)
        
        # Classification heads
        self.session_head = nn.Linear(hidden_dim * 3, num_sessions)
        self.trend_head = nn.Linear(hidden_dim * 3, num_trends)
    
    def forward(self, x_1s, x_1m, x_5m):
        # Embed each timeframe
        h_1s = self.embed_1s(x_1s)  # (batch, 60, hidden)
        h_1m = self.embed_1m(x_1m)  # (batch, 60, hidden)
        h_5m = self.embed_5m(x_5m)  # (batch, 78, hidden)
        
        # Apply mixer blocks
        h_1s = self.mixer_1s(h_1s)
        h_1m = self.mixer_1m(h_1m)
        h_5m = self.mixer_5m(h_5m)
        
        # Global average pooling + norm
        h_1s = self.norm_1s(h_1s.mean(dim=1))  # (batch, hidden)
        h_1m = self.norm_1m(h_1m.mean(dim=1))
        h_5m = self.norm_5m(h_5m.mean(dim=1))
        
        # Concatenate and classify
        combined = torch.cat([h_1s, h_1m, h_5m], dim=1)
        
        return self.session_head(combined), self.trend_head(combined)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
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
    total_loss = 0
    correct = 0
    total = 0
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
    train_path = 'data/train.parquet'
    test_path = 'data/test.parquet'
    batch_size = 256
    num_epochs = 1
    lr = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_ds = TradingDataset(train_path, window_size=60, stride=5)
    test_ds = TradingDataset(test_path, window_size=60, stride=5)
    print(f"Train: {len(train_ds):,}, Test: {len(test_ds):,}")
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=RowGroupSampler(train_ds, shuffle=True), num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        sampler=RowGroupSampler(test_ds, shuffle=False), num_workers=0
    )
    
    model = TradingMixer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['acc']:.4f}")
    
    torch.save(model.state_dict(), 'data/mixer_model.pt')
    print("\nSaved model to data/mixer_model.pt")


if __name__ == '__main__':
    main()
