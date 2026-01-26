"""Transformer model for trading trend classification using x-transformers."""

import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Encoder


class TradingTransformer(nn.Module):
    """
    Multi-timeframe Transformer classifier for trend prediction.
    
    Inputs:
        features_1s: (batch, 60, 4) - 1-second OHLC
        features_1m: (batch, 60, 4) - 1-minute OHLC  
        features_5m: (batch, 78, 4) - 5-minute OHLC
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
    ):
        super().__init__()
        
        # Separate transformer for each timeframe
        self.transformer_1s = ContinuousTransformerWrapper(
            dim_in=input_size,
            dim_out=d_model,
            max_seq_len=60,
            attn_layers=Encoder(
                dim=d_model,
                depth=num_layers,
                heads=nhead,
                rotary_pos_emb=True
            )
        )
        
        self.transformer_1m = ContinuousTransformerWrapper(
            dim_in=input_size,
            dim_out=d_model,
            max_seq_len=60,
            attn_layers=Encoder(
                dim=d_model,
                depth=num_layers,
                heads=nhead,
                rotary_pos_emb=True
            )
        )
        
        self.transformer_5m = ContinuousTransformerWrapper(
            dim_in=input_size,
            dim_out=d_model,
            max_seq_len=78,
            attn_layers=Encoder(
                dim=d_model,
                depth=num_layers,
                heads=nhead,
                rotary_pos_emb=True
            )
        )
        
        # Classification heads on concatenated representations
        self.session_head = nn.Linear(d_model * 3, num_sessions)
        self.trend_head = nn.Linear(d_model * 3, num_trends)
    
    def forward(self, x_1s: torch.Tensor, x_1m: torch.Tensor, x_5m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_1s: (batch, 60, 4) 1-second OHLC features
            x_1m: (batch, 391, 4) 1-minute OHLC features
            x_5m: (batch, 79, 4) 5-minute OHLC features
        Returns:
            session_logits: (batch, 3)
            trend_logits: (batch, 7)
        """
        h_1s = self.transformer_1s(x_1s)[:, -1, :]  # (batch, d_model)
        h_1m = self.transformer_1m(x_1m)[:, -1, :]  # (batch, d_model)
        h_5m = self.transformer_5m(x_5m)[:, -1, :]  # (batch, d_model)
        
        combined = torch.cat([h_1s, h_1m, h_5m], dim=1)  # (batch, d_model * 3)
        
        session_logits = self.session_head(combined)
        trend_logits = self.trend_head(combined)
        
        return session_logits, trend_logits


# Alias for backward compatibility
TradingLSTM = TradingTransformer
