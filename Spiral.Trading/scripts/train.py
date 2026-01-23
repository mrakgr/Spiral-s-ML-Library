#!/usr/bin/env python3
"""Training script for TradingLSTM model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TradingDataset, RowGroupSampler
from model import TradingLSTM


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    trend_correct = 0
    total = 0
    
    trend_criterion = nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(loader):
        features = batch['features'].to(device)
        trend_labels = batch['trend'].to(device)
        
        optimizer.zero_grad()
        session_logits, trend_logits = model(features)
        
        # Only train on trend prediction for now
        # Session prediction requires deeper context (1m/5m aggregates)
        loss = trend_criterion(trend_logits, trend_labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        trend_correct += (trend_logits.argmax(1) == trend_labels).sum().item()
        total += features.size(0)
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: loss={loss.item():.4f}")
    
    return {
        'loss': total_loss / len(loader),
        'trend_acc': trend_correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    trend_correct = 0
    total = 0
    
    trend_criterion = nn.CrossEntropyLoss()
    
    for batch in loader:
        features = batch['features'].to(device)
        trend_labels = batch['trend'].to(device)
        
        session_logits, trend_logits = model(features)
        
        loss = trend_criterion(trend_logits, trend_labels)
        
        total_loss += loss.item()
        trend_correct += (trend_logits.argmax(1) == trend_labels).sum().item()
        total += features.size(0)
    
    return {
        'loss': total_loss / len(loader),
        'trend_acc': trend_correct / total,
    }


def main():
    # Config
    train_path = 'data/train.parquet'
    test_path = 'data/test.parquet'
    window_size = 60
    stride = 5
    batch_size = 1024
    num_epochs = 1
    learning_rate = 1e-3
    num_workers = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading datasets...")
    train_dataset = TradingDataset(train_path, window_size=window_size, stride=stride)
    test_dataset = TradingDataset(test_path, window_size=window_size, stride=stride)
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=RowGroupSampler(train_dataset, shuffle=True), num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=RowGroupSampler(test_dataset, shuffle=False), num_workers=num_workers
    )
    
    # Model
    model = TradingLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch + 1}/{num_epochs} ===")
        
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Trend Acc: {train_metrics['trend_acc']:.4f}")
        
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, "
              f"Trend Acc: {test_metrics['trend_acc']:.4f}")
        print()
    
    # Save model
    torch.save(model.state_dict(), 'data/model.pt')
    print("Model saved to data/model.pt")


if __name__ == "__main__":
    main()
