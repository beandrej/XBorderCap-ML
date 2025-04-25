import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self):
        from config import SHARED_HIDDEN, DROPOUT, LEAKY_RELU
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(SHARED_HIDDEN, SHARED_HIDDEN),
            nn.BatchNorm1d(SHARED_HIDDEN),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(DROPOUT),
            nn.Linear(SHARED_HIDDEN, SHARED_HIDDEN),
            nn.BatchNorm1d(SHARED_HIDDEN),
        )
        self.activation = nn.LeakyReLU(LEAKY_RELU)

    def forward(self, x):
        return self.activation(self.block(x) + x)

class ResidualTCNBlock(nn.Module):
    def __init__(self, dilation):
        from config import SHARED_HIDDEN, DROPOUT, LEAKY_RELU, KERNEL_SIZE, STRIDE
        super().__init__()
        padding = (KERNEL_SIZE - 1) * dilation
        self.conv = nn.Conv1d(SHARED_HIDDEN, SHARED_HIDDEN, KERNEL_SIZE, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(SHARED_HIDDEN)
        self.relu = nn.LeakyReLU(LEAKY_RELU)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]
        out = self.bn(out)
        out = self.relu(out)
        return self.dropout(out + x)

class ResidualTCN(nn.Module):
    def __init__(self):
        super().__init__()
        from config import DILATION
        layers = []
        for dilation in DILATION:
            layers += [ResidualTCNBlock(dilation)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

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
    def __init__(self, input_dim, out_dim, task_type='classification', n_heads=4):
        from config import SHARED_HIDDEN, CLS_HIDDEN, REG_HIDDEN, DROPOUT
        super().__init__()

        self.task_type = task_type
        self.input_dim = input_dim
        self.hidden_dim = SHARED_HIDDEN

        self.input_proj = nn.Linear(input_dim, SHARED_HIDDEN)

        self.attn = nn.MultiheadAttention(embed_dim=SHARED_HIDDEN, num_heads=n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(SHARED_HIDDEN)

        self.shared = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        # Single regression head
        self.reg_head = nn.Sequential(
            nn.Linear(SHARED_HIDDEN, REG_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(REG_HIDDEN, REG_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(REG_HIDDEN // 2, 1)
        )

        # Single classification head
        self.cls_head = nn.Sequential(
            nn.Linear(SHARED_HIDDEN, CLS_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(CLS_HIDDEN, CLS_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(CLS_HIDDEN // 2, out_dim)
        )

    def forward(self, x):  # x: [batch, input_dim]
        x_proj = self.input_proj(x)          # [batch, hidden_dim]
        x_seq = x_proj.unsqueeze(1)          # [batch, 1, hidden_dim]

        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x_attn = self.attn_norm(attn_out + x_seq)
        x_flat = x_attn.squeeze(1)           # [batch, hidden_dim]

        x_shared = self.shared(x_flat)

        if self.task_type == 'classification':
            return self.cls_head(x_shared)
        else:
            return self.reg_head(x_shared)

class TCN(nn.Module):
    def __init__(self, input_dim, out_dim, task_type):
        super().__init__()
        from config import SHARED_HIDDEN, REG_HIDDEN, LEAKY_RELU, CLS_HIDDEN, DROPOUT

        self.task_type = task_type
        self.input_proj = nn.Linear(input_dim, SHARED_HIDDEN)
        self.tcn = ResidualTCN()

        self.reg_head = nn.Sequential(
            nn.Linear(SHARED_HIDDEN, REG_HIDDEN),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(DROPOUT),
            nn.Linear(REG_HIDDEN, REG_HIDDEN // 2),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(DROPOUT),
            nn.Linear(REG_HIDDEN // 2, out_dim)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(SHARED_HIDDEN, CLS_HIDDEN),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(DROPOUT),
            nn.Linear(CLS_HIDDEN, CLS_HIDDEN),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(DROPOUT),
            nn.Linear(CLS_HIDDEN, out_dim)
        )
    
    def forward(self, x):      # x: [batch, seq_len, input_dim]
        x = self.input_proj(x) # -> [batch, seq_len, HIDDEN]
        x = x.transpose(1, 2)  # -> [batch, HIDDEN, seq_len]
        x = self.tcn(x)        # -> [batch, HIDDEN, seq_len]
        x = x[:, :, -1]        # take last time step

        if self.task_type == 'classification':
            return self.cls_head(x)
        else:
            return self.reg_head(x)


def getModel(model_name, input_dim, output_dim):
    if model_name == "BaseModel":
        return BaseModel(input_dim, output_dim)
    elif model_name == "Net":
        return Net(input_dim, output_dim)
    elif model_name == "LSTM":
        return LSTM(input_dim, output_dim)
    elif model_name == "Hybrid":
        return Hybrid(input_dim, output_dim)
    elif model_name == "TCN":
        return TCN(input_dim, n_classes=output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
