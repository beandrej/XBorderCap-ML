import torch
import os
import pandas as pd
from tqdm import tqdm
from data_loader import CrossBorderData
from torch.utils.data import DataLoader
from model import *
import config

model_name = config.MODEL_NAME
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/model_params', f"{model_name}.pth")
pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/predictions', f"{model_name}.csv")

test_data = CrossBorderData(train=False)
test_tensor = test_data.to_tensor()
test_loader = DataLoader(test_data, batch_size=1) 

# Load trained model
input_dim = next(iter(test_loader))[0].shape[1]
model = LinReg(input_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Store predictions
predictions = []
timestamps = []

# Predict with tqdm progress bar
with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="Making Predictions", leave=True)
    
    for X_batch, _ in progress_bar:
        pred = model(X_batch)
        predictions.append(pred.item())
        # Update tqdm with current prediction
        progress_bar.set_postfix(current_pred=pred.item())

# Save predictions to CSV
pred_df = pd.DataFrame({
    "timestamp": pd.to_datetime(test_data.timestamp), 
    "predicted_capacity": predictions
})

pred_df.to_csv(pred_path, index=False)
print("Predictions saved. ✅")
