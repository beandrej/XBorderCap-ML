import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

"""
********************************* BUILDING BLOCKS ******************************************
"""

class ResidualBlock(nn.Module):
    def __init__(self):
        import config
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(config.SHARED_HIDDEN, config.SHARED_HIDDEN),
            nn.BatchNorm1d(config.SHARED_HIDDEN),
            nn.LeakyReLU(config.LEAKY_RELU),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.SHARED_HIDDEN, config.SHARED_HIDDEN),
            nn.BatchNorm1d(config.SHARED_HIDDEN),
        )
        self.activation = nn.LeakyReLU(config.LEAKY_RELU)

    def forward(self, x):
        return self.activation(self.block(x) + x)

class ResidualTCNBlock(nn.Module):
    def __init__(self, dilation):
        import config
        super().__init__()
        padding = (config.KERNEL_SIZE - 1) * dilation
        self.conv = nn.Conv1d(config.SHARED_HIDDEN, config.SHARED_HIDDEN, kernel_size=config.KERNEL_SIZE, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(config.SHARED_HIDDEN)
        self.relu = nn.LeakyReLU(config.LEAKY_RELU)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]
        out = self.bn(out)
        out = self.relu(out)
        return self.dropout(out + x)

class ResidualTCNBlockStride(nn.Module):
    def __init__(self):
        import config
        super().__init__()
        self.conv = nn.Conv1d(
            config.SHARED_HIDDEN, 
            config.SHARED_HIDDEN, 
            kernel_size=config.KERNEL_SIZE, 
            stride=config.KERNEL_SIZE,   
            padding=0,             
            dilation=1            
        )
        self.bn = nn.BatchNorm1d(config.SHARED_HIDDEN)
        self.relu = nn.LeakyReLU(config.LEAKY_RELU)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        stride = self.conv.stride[0]
        expected_len = out.shape[-1]
        residual = x[:, :, :expected_len * stride:stride]
        return self.dropout(out + residual)
    
class ResidualTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = ResidualTCNBlockStride()
        
    def forward(self, x):
        return self.network(x)

"""
********************************* MODELS ******************************************
"""

class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim, task_type='regression'):
        super(BaseModel, self).__init__()

        self.model_type = 'Reg'
        self.sequence = False
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class Net(nn.Module):
    def __init__(self, input_dim, output_dim, task_type='regression'):
        import config
        super().__init__()
        self.model_type = 'Reg'
        self.sequence = False
        self.input_proj = nn.Linear(input_dim, config.SHARED_HIDDEN)

        layers = []
        num_layers = 7
        for i in range(num_layers):
            layers.append(nn.Linear(config.SHARED_HIDDEN, config.SHARED_HIDDEN))
            layers.append(nn.LayerNorm(config.SHARED_HIDDEN))
            layers.append(nn.LeakyReLU(config.LEAKY_RELU))

            if i >= num_layers - 2:
                layers.append(nn.Dropout(config.DROPOUT))

        self.body = nn.Sequential(*layers)
        self.output_layer = nn.Linear(config.SHARED_HIDDEN, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.body(x)
        return self.output_layer(x)

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, task_type='regression', hidden_dim=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.model_type = 'Reg'
        self.sequence = True

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        direction_factor = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_dim * direction_factor)
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)              
        out = out[:, -1, :]                
        out = self.norm(out)              
        return self.fc(out)                
    
class Hybrid(nn.Module):
    def __init__(self, input_dim, out_dim, task_type, n_heads=4):
        import config
        super().__init__()
        self.model_type = 'Hybrid'
        self.sequence = False
        self.task_type = task_type
        self.input_dim = input_dim
        self.hidden_dim = config.SHARED_HIDDEN

        self.input_proj = nn.Linear(input_dim, config.SHARED_HIDDEN)

        self.attn = nn.MultiheadAttention(embed_dim=config.SHARED_HIDDEN, num_heads=n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(config.SHARED_HIDDEN)

        self.shared = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        # Single regression head
        self.reg_head = nn.Sequential(
            nn.Linear(config.SHARED_HIDDEN, config.REG_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.REG_HIDDEN, config.REG_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.REG_HIDDEN // 2, 1)
        )

        # Single classification head
        self.cls_head = nn.Sequential(
            nn.Linear(config.SHARED_HIDDEN, config.CLS_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.CLS_HIDDEN, config.CLS_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.CLS_HIDDEN // 2, out_dim)
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

class TCNHybrid(nn.Module):
    def __init__(self, input_dim, out_dim, task_type):
        super().__init__()
        import config
        self.model_type = 'Hybrid'
        self.sequence = True
        self.task_type = task_type
        self.input_proj = nn.Linear(input_dim, config.SHARED_HIDDEN)
        self.tcn = ResidualTCN()

        self.reg_head = nn.Sequential(
            nn.Linear(config.SHARED_HIDDEN, config.REG_HIDDEN),
            nn.LeakyReLU(config.LEAKY_RELU),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.REG_HIDDEN, config.REG_HIDDEN // 2),
            nn.LeakyReLU(config.LEAKY_RELU),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.REG_HIDDEN // 2, out_dim)
        )

        self.cls_head = nn.Sequential(
            nn.Linear(config.SHARED_HIDDEN, config.CLS_HIDDEN),
            nn.LeakyReLU(config.LEAKY_RELU),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.CLS_HIDDEN, config.CLS_HIDDEN),
            nn.LeakyReLU(config.LEAKY_RELU),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.CLS_HIDDEN, out_dim)
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

"""
********************************* UTILS ******************************************
"""

def getModel(model_name, input_dim, output_dim, task_type):
    model = loadModel(model_name, input_dim, output_dim, task_type)

    if isRegressionOnly(model):
        print("Model performing regression\n")
        task_type = 'regression'
        output_dim = 1
    else:
        print("Model with hybrid solver: Regression <-> Classification\n")

    return loadModel(model_name, input_dim, output_dim, task_type), task_type, output_dim

def isRegressionOnly(model):
    return model.model_type == 'Reg'
    
def loadModel(model_name, input_dim, output_dim, task_type):
    if model_name == "BaseModel":
        return BaseModel(input_dim, output_dim, task_type=task_type)
    elif model_name == "Net":
        return Net(input_dim, output_dim, task_type=task_type)
    elif model_name == "LSTM":
        return LSTM(input_dim, output_dim, task_type=task_type)
    elif model_name == "Hybrid":
        return Hybrid(input_dim, output_dim, task_type=task_type)
    elif model_name == "TCNHybrid":
        return TCNHybrid(input_dim, output_dim, task_type=task_type)
    else:
        raise ValueError(f"Unknown model: {model_name}")