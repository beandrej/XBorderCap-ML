import torch
import os
import pandas as pd
from tqdm import tqdm
from data_loader import CrossBorderData
from torch.utils.data import DataLoader
from model import *
import config

MODEL_NAME = config.MODEL_NAME
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/model_params', f"{MODEL_NAME}.pth")
pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/predictions', f"{MODEL_NAME}.csv")

test_data = CrossBorderData(config.C1, config.C2, config.DOMAIN, train=False)
test_tensor = test_data.to_tensor()
test_loader = DataLoader(test_data, batch_size=1) 

input_dim = next(iter(test_loader))[0].shape[1]
model = get_model(MODEL_NAME, input_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

predictions = []
timestamps = []


with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="Making Predictions", leave=True)
    
    for X_batch, _ in progress_bar:

        if MODEL_NAME == "lstm":
            X_batch = X_batch.view(X_batch.shape[0], 1, X_batch.shape[1])

        pred = model(X_batch)
        predictions.append(pred.item())
        progress_bar.set_postfix(current_pred=pred.item())


pred_df = pd.DataFrame({
    "timestamp": pd.to_datetime(test_data.timestamp), 
    "predicted_capacity": predictions
})

pred_df.to_csv(pred_path, index=False)
print("Predictions saved. ✅")
