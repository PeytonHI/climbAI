# models.py
import torch
import torch.nn as nn
import math
import importlib

class PoseTransformer(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1, cond_dim=0):
        """
        input_dim: per-time-step flattened pose dimension (33*3 = 99 if using xy+vis)
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.cond_dim = cond_dim
        # if conditioning vector is present, project and concatenate to input
        if cond_dim and cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, d_model)
            self.input_proj = nn.Linear(input_dim + d_model, d_model)
        else:
            self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = DynamicPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # decoder: predict full sequence autoregressively / or use decoder to map
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: if conditioning used, expect tuple (poses, cond) where cond is (B, cond_dim)
        if isinstance(x, tuple) or isinstance(x, list):
            poses, cond = x
            # cond: (B, cond_dim) -> project -> (B, d_model) -> expand to (B,T,d_model)
            c = self.cond_proj(cond)  # (B,d_model)
            B, T, _ = poses.shape
            c_exp = c.unsqueeze(1).expand(-1, T, -1)
            inp = torch.cat([poses, c_exp], dim=-1)
            h = self.input_proj(inp)
        else:
            # x: (B, T, input_dim)
            h = self.input_proj(x)  # (B,T,d_model)
        h = self.pos_enc(h)
        h = self.encoder(h)  # (B,T,d_model)
        out = self.output_proj(h)  # (B,T,input_dim)
        return out


class SimpleImageEncoder(nn.Module):
    """
    Small image encoder to produce a fixed-size vector from an RGB image tensor (B,3,H,W).
    If torchvision and resnet18 are available, use a pretrained ResNet18 (without final fc).
    Otherwise use a small 3-layer convnet.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.use_resnet = False
        if importlib.util.find_spec('torchvision') is not None:
            try:
                import torchvision.models as models
                resnet = models.resnet18(pretrained=True)
                # remove fc
                modules = list(resnet.children())[:-1]
                self.backbone = nn.Sequential(*modules)
                self.pool_out = True
                self.fc = nn.Linear(512, out_dim)
                self.use_resnet = True
            except Exception:
                self._make_small_cnn()
        else:
            self._make_small_cnn()

    def _make_small_cnn(self):
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.pool_out = False
        self.fc = nn.Linear(64, self.out_dim)

    def forward(self, x):
        # x: (B,3,H,W)
        h = self.backbone(x)
        if self.pool_out:
            h = h.view(h.size(0), -1)
        else:
            h = h.view(h.size(0), -1)
        return self.fc(h)

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