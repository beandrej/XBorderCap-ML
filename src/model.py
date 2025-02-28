import torch
import torch.nn as nn
import config

class LinReg(nn.Module):
    def __init__(self, input_dim):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.3) 
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x = self.batch_norm(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return torch.relu(self.fc3(x)) 

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT):
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
    if model_name == "linreg":
        return LinReg(input_dim)
    elif model_name == "nn":
        return Net(input_dim)
    elif model_name == "lstm":
        return LSTM(input_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")