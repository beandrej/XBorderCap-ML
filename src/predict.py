import os
import torch
import pandas as pd
from tqdm import tqdm
from data_loader import CrossBorderData
from torch.utils.data import DataLoader, Subset
from model import *
import config
import data_loader

MODEL_NAME = config.MODEL_NAME
SPLIT_RATIO = config.TRAIN_SPLIT

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_params', f"{MODEL_NAME}.pth")
pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions', f"{MODEL_NAME}_{data_loader.DATASET_NAME}_{data_loader.DOMAIN}.csv")

full_df = CrossBorderData(data_loader.COUNTRY1, data_loader.COUNTRY2, data_loader.DOMAIN, data_loader.DATASET_NAME, load_from_file=True)
split_index = int(len(full_df) * SPLIT_RATIO)

test_df = Subset(full_df, range(split_index, len(full_df)))
test_loader = DataLoader(test_df, batch_size=1, shuffle=False)

test_indices = range(split_index, len(full_df))
test_timestamps = [full_df[i][2] for i in test_indices]
test_timestamps = [str(ts) for ts in test_timestamps]

input_dim = next(iter(test_loader))[0].shape[1]
model = get_model(MODEL_NAME, input_dim)
print(f"\nLoading model: {MODEL_NAME}\n")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

predictions = []
timestamps = []


with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="Making Predictions", leave=True)
    
    for X_batch, _, _ in progress_bar:

        if MODEL_NAME == "lstm":
            X_batch = X_batch.view(X_batch.shape[0], 1, X_batch.shape[1])

        pred = model(X_batch)
        predictions.append(pred.item())
        progress_bar.set_postfix(current_pred=pred.item())


pred_df = pd.DataFrame({
    "timestamp": pd.to_datetime(test_timestamps), 
    "predicted_capacity": predictions
})

pred_df.to_csv(pred_path, index=False)
print("Predictions saved")
