# models.py
import torch
import torch.nn as nn
import math

class PoseTransformer(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        """
        input_dim: per-time-step flattened pose dimension (33*3 = 99 if using xy+vis)
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = DynamicPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # decoder: predict full sequence autoregressively / or use decoder to map
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (B, T, input_dim)
        h = self.input_proj(x)  # (B,T,d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)  # (B,T,d_model)
        out = self.output_proj(h)  # (B,T,input_dim)
        return out

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=1000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # (1,max_len,d_model)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x: (B,T,d_model)
#         T = x.shape[1]
#         return x + self.pe[:, :T, :]

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, D = x.shape
        assert D == self.d_model, "Input dimension mismatch"

        position = torch.arange(T, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=x.device).float() * (-math.log(10000.0) / D))

        pe = torch.zeros(T, D, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        return x + pe