#!/usr/bin/env python3
"""Test the TradingLSTM model."""

import torch
from model import TradingLSTM


def main():
    print("=== Model Architecture ===")
    model = TradingLSTM()
    print(model)
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    print("=== Forward Pass Test ===")
    batch_size = 32
    seq_len = 60
    x = torch.randn(batch_size, seq_len, 4)
    
    session_logits, trend_logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Session logits shape: {session_logits.shape}")
    print(f"Trend logits shape: {trend_logits.shape}")
    print()
    
    print("=== CUDA Available ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        model = model.cuda()
        x = x.cuda()
        session_logits, trend_logits = model(x)
        print("Forward pass on GPU: OK")


if __name__ == "__main__":
    main()
