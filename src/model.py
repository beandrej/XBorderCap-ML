import torch
import torch.nn as nn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(input_dim, input_dim)
        self.batch_norm2 = nn.BatchNorm1d(input_dim)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(input_dim, input_dim)
        self.batch_norm3 = nn.BatchNorm1d(input_dim)

        self.fc4 = nn.Linear(input_dim, input_dim)
        self.batch_norm4 = nn.BatchNorm1d(input_dim)

        self.fc5 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.batch_norm3(x)

        x = torch.nn.functional.leaky_relu(self.fc4(x), negative_slope=0.01)
        x = self.batch_norm4(x)

        x = self.fc5(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.15):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional
        )
        self.attn = nn.Linear(hidden_dim * self.num_directions, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        B, T, D = x.size()

        h0 = torch.zeros(self.num_layers * self.num_directions, B, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, B, self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: [B, T, hidden_dim * num_directions]

        attn_weights = self.attn(out)              # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # softmax over time
        context = torch.sum(attn_weights * out, dim=1)     # [B, H]

        out = self.output(context)  # [B, output_dim]

        return out

class HybridOutputMLP(nn.Module):
    def __init__(self, input_dim, cls_dims, reg_count, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3)
        )
        self.cls_heads = nn.ModuleList([nn.Linear(hidden_dim, n) for n in cls_dims])
        self.reg_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(reg_count)])

    def forward(self, x):
        x = self.shared(x)
        cls_outputs = [head(x) for head in self.cls_heads]
        reg_outputs = [head(x) for head in self.reg_heads]
        return cls_outputs, reg_outputs


class XGBoostClassifierWrapper:
    def __init__(self, input_dim, output_dim, seed=42):
        base_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            tree_method="gpu_hist",
            predictor="gpu_predictor"
        )
        self.model = MultiOutputClassifier(base_model)
        self.fitted = False

    def fit(self, X_train, Y_train):
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(Y_train, torch.Tensor):
            Y_train = Y_train.cpu().numpy()
        self.model.fit(X_train, Y_train)
        self.fitted = True

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict(X)

    def evaluate(self, X_val, Y_val):
        Y_pred = self.predict(X_val)
        if isinstance(Y_val, torch.Tensor):
            Y_val = Y_val.cpu().numpy()
        acc = accuracy_score(Y_val, Y_pred)
        report = classification_report(Y_val, Y_pred, output_dict=True)
        return {"accuracy": acc, "report": report}



def get_model(model_name, input_dim, output_dim):
    print(f"\nUsing model: {model_name}")
    if model_name == "BaseModel":
        return BaseModel(input_dim, output_dim)
    elif model_name == "Net":
        return Net(input_dim, output_dim)
    elif model_name == "LSTM":
        return LSTM(input_dim, output_dim)
    elif model_name == "XGBClass":
        return XGBoostClassifierWrapper(input_dim, output_dim)
    elif model_name == "MLP":
        return HybridOutputMLP(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
