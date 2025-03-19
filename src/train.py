import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import config
from data_class import *
import data_loader
from model import *
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_to_load = "BASELINE_MAXBEX"

MODEL_NAME = config.MODEL_NAME
SPLIT_RATIO = config.TRAIN_SPLIT
VALID_SPLIT = config.VALID_SPLIT

full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{df_to_load}.csv"), index_col=0)
first_target_idx = full_df.columns.get_loc("AUS_CZE")

X = full_df.iloc[:, :first_target_idx]
Y = full_df.iloc[:, first_target_idx:]

split_index = int(len(full_df) * SPLIT_RATIO)
val_index = int(split_index * (1 - VALID_SPLIT))

X_train_full = X.iloc[:split_index]
Y_train_full = Y.iloc[:split_index]

X_train = X_train_full.iloc[:val_index]
Y_train = Y_train_full.iloc[:val_index]

X_val = X_train_full.iloc[val_index:]
Y_val = Y_train_full.iloc[val_index:]

print("Shape of X_train:", X_train.shape) 
print("Shape of Y_train:", Y_train.shape) 
print("Shape of X_val:", X_val.shape) 
print("Shape of Y_val:", Y_val.shape) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train.values, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(Y_val.values, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

input_dim = next(iter(train_loader))[0].shape[1]
output_dim = Y_train.shape[1]
model = get_model(MODEL_NAME, input_dim, output_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print("Model parameters:\n", f"Train split: {config.TRAIN_SPLIT}\n", f"Validation split: {config.VALID_SPLIT}\n",
      f"Batch size: {config.BATCH_SIZE}\n", f"Learning rate: {config.LEARNING_RATE}")


train_losses = []
val_losses = []
r2_scores = []
mae_scores = []

for epoch in range(config.EPOCHS):
    model.train()
    epoch_loss = 0  
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=True)
    
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

    model.eval()
    val_loss = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if MODEL_NAME == "lstm":
                X_batch = X_batch.view(X_batch.shape[0], 1, X_batch.shape[1])
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()

            y_true_list.append(y_batch.cpu())
            y_pred_list.append(predictions.cpu())
    
    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    r2_scores.append(r2)
    mae_scores.append(mae)

    if epoch % 2 == 0:
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss (MSE): {train_losses[-1]:.2f} | Val Loss (MSE): {val_losses[-1]:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}")
    scheduler.step()

epochs_list = list(range(1, config.EPOCHS + 1))
metrics_df = pd.DataFrame({
    'epoch': epochs_list,
    'train_loss': train_losses,
    'val_loss': val_losses,
    'r2': r2_scores,
    'mae': mae_scores
})

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_metrics', f"{config.MODEL_NAME}_{df_to_load}_metrics.csv")
metrics_df.to_csv(csv_path, index=False)
print(f"Metrics saved to: {csv_path}")

torch_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_params', f"{MODEL_NAME}.pth")
torch.save(model.state_dict(), torch_model_path)
print(f"Model saved at: {torch_model_path}")
