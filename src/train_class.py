import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import pandas as pd
import numpy as np
import config
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score, mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_to_load = "BASELINE_NTC"
SPLIT_RATIO = config.TRAIN_SPLIT
VALID_SPLIT = config.VALID_SPLIT

# Load Data
full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{df_to_load}.csv"), index_col=0)

# Identify feature and target columns
first_target_idx = full_df.columns.get_loc("AT_to_IT_NORD")  # Adjust to first border col
features = full_df.iloc[:, :first_target_idx]
targets = full_df.iloc[:, first_target_idx:]

borders = targets.columns.tolist()

# LabelEncode each border separately
label_encoders = {}
border_class_counts = {}
for border in borders:
    le = LabelEncoder()
    targets[border] = le.fit_transform(targets[border])
    label_encoders[border] = le
    border_class_counts[border] = len(le.classes_)
print(f"Borders and class counts: {border_class_counts}")

# Train / Validation Split
split_index = int(len(full_df) * SPLIT_RATIO)
val_index = int(split_index * (1 - VALID_SPLIT))

X_train_full = features.iloc[:split_index]
Y_train_full = targets.iloc[:split_index]

X_train = X_train_full.iloc[:val_index]
Y_train = Y_train_full.iloc[:val_index]

X_val = X_train_full.iloc[val_index:]
Y_val = Y_train_full.iloc[val_index:]

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Torch Datasets
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train.values, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(Y_val.values, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Multi-Head Model
class MultiBorderClassifier(nn.Module):
    def __init__(self, input_dim, border_class_counts):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(config.DROPOUT_NN),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(config.DROPOUT_NN)
        )
        self.heads = nn.ModuleDict({
            border: nn.Linear(256, num_classes)
            for border, num_classes in border_class_counts.items()
        })

    def forward(self, x):
        shared_out = self.shared(x)
        outputs = {border: head(shared_out) for border, head in self.heads.items()}
        return outputs

# Initialize model
input_dim = X_train.shape[1]
model = MultiBorderClassifier(input_dim, border_class_counts).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print("Starting Multi-Border Classifier Training...")

for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = 0
        for border_idx, border in enumerate(borders):
            loss += criterion(outputs[border], Y_batch[:, border_idx])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)

    # Validation
    model.eval()
    val_loss = 0
    border_preds = {border: [] for border in borders}
    border_trues = {border: [] for border in borders}

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            batch_loss = 0
            for border_idx, border in enumerate(borders):
                batch_loss += criterion(outputs[border], Y_batch[:, border_idx])
                preds = torch.argmax(outputs[border], dim=1).cpu().numpy()
                border_preds[border].extend(preds)
                border_trues[border].extend(Y_batch[:, border_idx].cpu().numpy())
            val_loss += batch_loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

# Evaluation Report Per Border
report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_metrics')
os.makedirs(report_dir, exist_ok=True)

for border in borders:
    acc = accuracy_score(border_trues[border], border_preds[border])
    report = classification_report(border_trues[border], border_preds[border], target_names=[str(cls) for cls in label_encoders[border].classes_])
    print(f"\nBorder: {border} | Accuracy: {acc:.4f}\n{report}")
    with open(os.path.join(report_dir, f"CLASSIFIER_REPORT_{border}.txt"), 'w') as f:
        f.write(report)

# Save Model
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_params', f"multi_border_classifier.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Multi-Border Classifier saved to: {model_save_path}")
