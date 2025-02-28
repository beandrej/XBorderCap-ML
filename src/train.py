import pandas as pd
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#from pycaret.regression import *
from model import LinReg
import config
from data_loader import CrossBorderData

model_name = "linreg"
batch_size = config.BATCH_SIZE

train_dataset = CrossBorderData(train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

input_dim = next(iter(train_loader))[0].shape[1]
model =  LinReg(input_dim=input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
epochs = config.EPOCHS

for epoch in range(epochs):
    epoch_loss = 0  # Track total loss for the epoch
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        # Update tqdm progress bar description with loss info
        progress_bar.set_postfix(loss=loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {epoch_loss/len(train_loader):.6f}")

# Save trained model
path = path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/torch_models', f"{model_name}.pth")
torch.save(model.state_dict(), path)
print("Model training complete. Saved model.")  
