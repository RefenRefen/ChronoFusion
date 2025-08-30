# src/chronofusion/models/transformer.py

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class SingleScaleEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True  # ✅ Now everything is (batch, seq_len, embed_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        seq_len = x.size(1)
        device = x.device

        # Causal mask: (seq_len, seq_len)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )

        out = self.transformer_encoder(x, mask=causal_mask)  # Now batch_first=True

        return out

class MultiScaleTransformer(nn.Module):
    def __init__(
        self,
        input_dim_low,
        input_dim_high,
        d_model,
        nhead,
        num_layers,
        output_dim=100
    ):
        super().__init__()
        self.low_encoder = SingleScaleEncoder(input_dim_low, d_model, nhead, num_layers)
        self.high_encoder = SingleScaleEncoder(input_dim_high, d_model, nhead, num_layers)

        # Fusion: concat both representations per timestep + projection to output_dim
        self.fusion_proj = nn.Linear(d_model * 2, output_dim)

    def forward(self, x_low, x_high):
        """
        x_low: (batch_size, seq_len, input_dim_low)  # e.g., 30min timeframe window
        x_high: (batch_size, seq_len, input_dim_high) # e.g., 5min timeframe window
        """
        enc_low = self.low_encoder(x_low)    # (batch_size, seq_len, d_model)
        enc_high = self.high_encoder(x_high) # (batch_size, seq_len, d_model)

        combined = torch.cat([enc_low, enc_high], dim=-1)  # (batch_size, seq_len, d_model*2)
        out = self.fusion_proj(combined)                   # (batch_size, seq_len, output_dim=100)

        return out
