import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import config
from data_class import CrossBorderData
from data_loader import DATASET_NAME
from model import get_model
from sklearn.model_selection import train_test_split
import pandas as pd

# **Set device**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_NAME = config.MODEL_NAME

# **Step 1: Load Dataset**
full_dataset = CrossBorderData("AUS", "BEL", "max_bex", DATASET_NAME, load_from_file=True)

# **Step 2: Convert Tensors Back to DataFrame**
df = pd.DataFrame(full_dataset.X.numpy(), columns=full_dataset.feature_columns)
df["cross_border_capacity"] = full_dataset.y.numpy().flatten()
df.index = pd.to_datetime(full_dataset.timestamp)

# **Step 3: Apply Time-Based Split**
train_data = df[df.index < "2024-01-01"]
test_data = df[df.index >= "2024-01-01"]

# **Step 4: Separate Features (X) and Target (y)**
X_train = train_data.drop(columns=["cross_border_capacity"])
y_train = train_data["cross_border_capacity"]

X_test = test_data.drop(columns=["cross_border_capacity"])
y_test = test_data["cross_border_capacity"]

# **Convert to PyTorch Tensors**
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# **Create DataLoaders**
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)

# **Step 5: Initialize Model**
input_dim = X_train.shape[1]
model = get_model(MODEL_NAME, input_dim).to(device)

# **Define Loss & Optimizer**
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# **Step 6: Train on Time-Based Split Data**
print("\n🎯 Training on Time-Based Split Data...")
for epoch in range(config.EPOCHS):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{config.EPOCHS}", leave=True)

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        if MODEL_NAME == "lstm":
            X_batch = X_batch.view(X_batch.shape[0], 1, X_batch.shape[1])

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)

    scheduler.step()
    print(f"Train Epoch {epoch+1}/{config.EPOCHS} - Avg Loss: {epoch_loss/len(train_loader):.6f}")

# **Step 7: Evaluate on Test Set**
model.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        if MODEL_NAME == "lstm":
            X_batch = X_batch.view(X_batch.shape[0], 1, X_batch.shape[1])

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        test_loss += loss.item()

print(f"\n📊 Final Test Loss: {test_loss/len(test_loader):.6f}")

# **Step 8: Save Model**
torch_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_params', f"{MODEL_NAME}.pth")
torch.save(model.state_dict(), torch_model_path)
print(f"✅ Model saved at: {torch_model_path}")
