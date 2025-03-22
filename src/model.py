import torch
import torch.nn as nn
import config

class Reg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Reg, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=config.DROPOUT_NN)

        self.fc2 = nn.Linear(1024, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=config.DROPOUT_NN)

        self.fc3 = nn.Linear(512, 256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(p=config.DROPOUT_NN)

        self.fc4 = nn.Linear(256, 128)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(p=config.DROPOUT_NN)

        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.batch_norm3(x)
        x = self.dropout3(x)

        x = torch.nn.functional.leaky_relu(self.fc4(x), negative_slope=0.01)
        x = self.batch_norm4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        return x

class MultiBorderClassifier(nn.Module):
    def __init__(self, input_dim, border_class_counts):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(config.DROPOUT_NN),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(config.DROPOUT_NN)
        )
        # Dynamically create classifier heads per border
        self.border_heads = nn.ModuleDict({
            border: nn.Linear(256, num_classes)
            for border, num_classes in border_class_counts.items()
        })

    def forward(self, x):
        shared_out = self.shared_fc(x)
        outputs = {}
        for border, head in self.border_heads.items():
            outputs[border] = head(shared_out)  # Raw logits for each border
        return outputs

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
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

def get_model(model_name, input_dim, output_dim):
    print(f"\nUsing model: {model_name}")
    if model_name == "BaseModel":
        return Reg(input_dim, output_dim)
    elif model_name == "nn":
        return Net(input_dim, output_dim)
    elif model_name == "lstm":
        return LSTM(input_dim, output_dim)
    elif model_name == "classifier":
        return MultiBorderClassifier(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
