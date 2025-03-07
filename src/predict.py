import os
import torch
import pandas as pd
from tqdm import tqdm
from data_loader import CrossBorderData
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from model import *
import config
import data_loader

MODEL_NAME = config.MODEL_NAME
SPLIT_RATIO = config.TRAIN_SPLIT

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/model_params', f"{MODEL_NAME}.pth")
pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions', f"{MODEL_NAME}_{data_loader.DATASET_NAME}_{data_loader.DOMAIN}.csv")

full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', "MAX_BEX_WITH_FEATURES.csv"), index_col=0)
first_target_idx = full_df.columns.get_loc("AUS_BEL")
split_index = int(len(full_df) * SPLIT_RATIO)

X = full_df.iloc[:, :first_target_idx]
Y = full_df.iloc[:, first_target_idx:]

split_index = int(len(full_df) * SPLIT_RATIO)
X_test = X.iloc[split_index:]
Y_test = Y.iloc[split_index:]

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

test_indices = range(split_index, len(full_df))
test_timestamps = full_df.index[test_indices]  # ✅ Extracts timestamps from index
test_timestamps = pd.to_datetime(test_timestamps)  # Ensure datetime format


input_dim = next(iter(test_loader))[0].shape[1]
output_dim = Y_test.shape[1]

model = get_model(MODEL_NAME, input_dim, output_dim)
print(f"\nLoading model: {MODEL_NAME}\n")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

predictions = []
timestamps = []


with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="Making Predictions", leave=True)
    
    for X_batch, _ in progress_bar:

        if MODEL_NAME == "lstm":
            X_batch = X_batch.view(X_batch.shape[0], 1, X_batch.shape[1])

        pred = model(X_batch)
        predictions.extend(pred.cpu().numpy().tolist())  # ✅ Keeps batch structure
        progress_bar.set_postfix(current_pred=pred[0, 0].item())  # ✅ Extracts first value from first row


border_names = Y_test.columns.to_list()
border_names = [f"{name}_pred" for name in border_names]

pred_df = pd.DataFrame(predictions, columns=border_names)
pred_df.insert(0, "timestamp", pd.to_datetime(test_timestamps))

pred_df.to_csv(pred_path, index=False)
print("Predictions saved")
