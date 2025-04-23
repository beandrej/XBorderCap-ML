import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.03),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.LeakyReLU(0.03)

    def forward(self, x):
        return self.activation(self.block(x) + x)

class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        from config import HIDDEN_DIM, DROPOUT
        super().__init__()
        self.input_proj = nn.Linear(input_dim, HIDDEN_DIM)

        layers = []
        num_layers = 7
        for i in range(num_layers):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            layers.append(nn.LayerNorm(HIDDEN_DIM))
            layers.append(nn.GELU())

            if i >= num_layers - 2:
                layers.append(nn.Dropout(DROPOUT))

        self.body = nn.Sequential(*layers)
        self.output_layer = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.body(x)
        return self.output_layer(x)

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, bidirectional=True):
        from config import HIDDEN_DIM, DROPOUT
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM * 2)
        self.fc = nn.Linear(
            HIDDEN_DIM * (2 if bidirectional else 1),
            output_dim
        )

    def forward(self, x, lengths):
        # x: (B, T, input_dim)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Use mean over time (ignoring padded parts)
        mask = torch.arange(out.size(1), device=x.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        out_avg = (out * mask).sum(dim=1) / lengths.unsqueeze(1)
        out_avg = self.norm(out_avg)

        return self.fc(out_avg)

class Hybrid(nn.Module):
    def __init__(self, input_dim, cls_dims, reg_count, n_heads=4):
        from config import HIDDEN_DIM, DROPOUT
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = HIDDEN_DIM

        self.input_proj = nn.Linear(input_dim, HIDDEN_DIM)

        self.attn = nn.MultiheadAttention(embed_dim=HIDDEN_DIM, num_heads=n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(HIDDEN_DIM)

        self.shared = nn.Sequential(
            ResidualBlock(HIDDEN_DIM, DROPOUT),
            ResidualBlock(HIDDEN_DIM, DROPOUT),
            ResidualBlock(HIDDEN_DIM, DROPOUT)
        )

        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM // 2, out_dim)
            ) for out_dim in cls_dims
        ])

        # Regression heads
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM // 2, 1)
            ) for _ in range(reg_count)
        ])

    def forward(self, x):
        x_proj = self.input_proj(x)
        x_seq = x_proj.unsqueeze(1)

        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x_attn = self.attn_norm(attn_out + x_seq)

        x_flat = x_attn.squeeze(1)

        x_shared = self.shared(x_flat)

        cls_outputs = [head(x_shared) for head in self.cls_heads]
        reg_outputs = [head(x_shared) for head in self.reg_heads]

        return cls_outputs, reg_outputs

# TODO build V2

class V2(nn.Module):
    def __init__(self, input_dim, hidden, n_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)

        # Temporal Convolution Stack (Dilated 1D CNN)
        self.temporal_blocks = nn.ModuleList()
        for dilation in [1, 2, 4, 7, 24]:
            self.temporal_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU()
                )
            )

        # Residual fusion
        self.fusion_proj = nn.Linear(hidden * len(self.temporal_blocks), 128)

        # Transformer encoder block (optional)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Latent mixing
        self.latent_mixer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Output heads
        self.cls_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.size()
        x = self.input_proj(x)  # (B, T, hidden)
        x = x.transpose(1, 2)  # (B, hidden, T)

        features = []
        for block in self.temporal_blocks:
            out = block(x)
            out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
            features.append(out)

        h = torch.cat(features, dim=-1)  # (B, hidden * #blocks)
        h = self.fusion_proj(h).unsqueeze(1)  # (B, 1, 128)

        # Transformer for temporal attention
        h = self.transformer(h)  # (B, 1, 128)
        h = h.squeeze(1)  # (B, 128)

        h = self.latent_mixer(h)

        return self.cls_head(h), self.reg_head(h)


def getModel(model_name, input_dim, output_dim):
    if model_name == "BaseModel":
        return BaseModel(input_dim, output_dim)
    elif model_name == "Net":
        return Net(input_dim, output_dim)
    elif model_name == "LSTM":
        return LSTM(input_dim, output_dim)
    elif model_name == "Hybrid":
        return Hybrid(input_dim, output_dim)
    elif model_name == "V2":
        return V2(input_dim, n_classes=output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
