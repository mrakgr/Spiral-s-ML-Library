"""LSTM model for trading session/trend classification."""

import torch
import torch.nn as nn


class TradingLSTM(nn.Module):
    """
    LSTM classifier with dual heads for session and trend prediction.
    
    Input: (batch, seq_len, 4) - OHLC returns
    Output: session logits (batch, 3), trend logits (batch, 7)
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_sessions: int = 3,
        num_trends: int = 7,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.session_head = nn.Linear(hidden_size, num_sessions)
        self.trend_head = nn.Linear(hidden_size, num_trends)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, 4) OHLC features
        Returns:
            session_logits: (batch, 3)
            trend_logits: (batch, 7)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        
        # Classification heads
        session_logits = self.session_head(last_hidden)
        trend_logits = self.trend_head(last_hidden)
        
        return session_logits, trend_logits
