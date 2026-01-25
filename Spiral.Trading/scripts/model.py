"""Transformer model for trading trend classification using x-transformers."""

import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Encoder


class TradingTransformer(nn.Module):
    """
    Transformer classifier for trend prediction.
    
    Input: (batch, seq_len, 4) - OHLC returns
    Output: session logits (batch, 3), trend logits (batch, 7)
    """
    
    def __init__(
        self,
        input_size: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_sessions: int = 3,
        num_trends: int = 7,
        max_seq_len: int = 60,
    ):
        super().__init__()
        
        self.transformer = ContinuousTransformerWrapper(
            dim_in=input_size,
            dim_out=d_model,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(
                dim=d_model,
                depth=num_layers,
                heads=nhead,
                rotary_pos_emb = True  # turns on rotary positional embeddings
            )
        )
        
        self.session_head = nn.Linear(d_model, num_sessions)
        self.trend_head = nn.Linear(d_model, num_trends)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, 4) OHLC features
        Returns:
            session_logits: (batch, 3)
            trend_logits: (batch, 7)
        """
        x = self.transformer(x)  # (batch, seq_len, d_model)
        last_hidden = x[:, -1, :]  # (batch, d_model)
        
        session_logits = self.session_head(last_hidden)
        trend_logits = self.trend_head(last_hidden)
        
        return session_logits, trend_logits


# Alias for backward compatibility
TradingLSTM = TradingTransformer
