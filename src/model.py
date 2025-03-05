import torch
import torch.nn as nn
import config

class LinReg(nn.Module):
    def __init__(self, input_dim):
        super(LinReg, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x @ self.weight + self.bias

class Reg(nn.Module):
    def __init__(self, input_dim):
        super(Reg, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=config.DROPOUT_NN)
        
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = torch.relu(self.fc2(x))
        x = self.batch_norm2(x)
        
        x = torch.relu(self.fc3(x))
        return torch.relu(self.fc4(x))


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT_LSTM):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        batch_size, seq_length, num_features = x.shape  
        x = x.view(batch_size * seq_length, num_features) 
        x = self.batch_norm(x)  
        x = x.view(batch_size, seq_length, num_features)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def get_model(model_name, input_dim):
    print(f"\nUsing model: {model_name}")
    if model_name == "linreg":
        return LinReg(input_dim)
    elif model_name == "reg":
        return Reg(input_dim)
    elif model_name == "nn":
        return Net(input_dim)
    elif model_name == "lstm":
        return LSTM(input_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
