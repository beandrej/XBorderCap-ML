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
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_NAME = config.MODEL_NAME
SPLIT_RATIO = config.TRAIN_SPLIT

full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "BASELINE_MAXBEX.csv"), index_col=0)
first_target_idx = full_df.columns.get_loc("AUS_CZE")

X = full_df.iloc[:, :first_target_idx]
Y = full_df.iloc[:, first_target_idx:]

split_index = int(len(full_df) * SPLIT_RATIO)
X_train = X.iloc[:split_index]
Y_train = Y.iloc[:split_index]

print("Shape of X_train:", X_train.shape) 
print("Shape of Y_train:", Y_train.shape) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

input_dim = next(iter(train_loader))[0].shape[1]
output_dim = Y_train.shape[1]
model = get_model(MODEL_NAME, input_dim, output_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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

    scheduler.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch+1}/{config.EPOCHS} - Avg Loss: {epoch_loss/len(train_loader):.6f}")

torch_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_params', f"{MODEL_NAME}.pth")
torch.save(model.state_dict(), torch_model_path)
print(f"Model saved at: {torch_model_path}")
