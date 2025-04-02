import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from model import HybridOutputMLP
from train_reg import TRAIN_SPLIT, BATCH_SIZE
import json

# --- CONFIG ---
MODEL_NAME = 'Hybrid'
BORDER_TYPE = 'NTC'
LOSS = 'Hybrid'
TRAINING_SET = 'BL_NTC_FULL'
UNIQUE_VAL_TRSH = 50

# --- PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "../prep_data", f"{TRAINING_SET}.csv")
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../model_params/{BORDER_TYPE}/{MODEL_NAME}', f"{MODEL_NAME}_{TRAINING_SET}_{LOSS}.pth")
mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../model_params/{BORDER_TYPE}/{MODEL_NAME}/mappings', f"cls_map_{MODEL_NAME}_{TRAINING_SET}_{UNIQUE_VAL_TRSH}.json")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../model_params/{BORDER_TYPE}/{MODEL_NAME}', f"params_{MODEL_NAME}_{TRAINING_SET}.json")
pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions_csv', f"pred_{MODEL_NAME}_{TRAINING_SET}_{LOSS}.csv")

# --- LOAD DATA ---
full_df = pd.read_csv(data_path, index_col=0)

# Split X and Y
if BORDER_TYPE == "MAXBEX":
    first_target_idx = full_df.columns.get_loc("AUS_CZE")
elif BORDER_TYPE == "NTC":
    first_target_idx = full_df.columns.get_loc("AT_to_IT_NORD")
else:
    raise ValueError("Wrong BORDER_TYPE!")

X = full_df.iloc[:, :first_target_idx]
Y = full_df.iloc[:, first_target_idx:]

# Create test split
split_index = int(len(full_df) * TRAIN_SPLIT)
X_test = X.iloc[split_index:]
Y_test = Y.iloc[split_index:]

# --- LOAD MODEL CONFIG + CLASS MAPPINGS ---
with open(config_path, "r") as f:
    config = json.load(f)

input_dim = config["input_dim"]
cls_dims = config["cls_dims"]
reg_count = config["reg_count"]
cls_cols = config["cls_cols"]
reg_cols = config["reg_cols"]

with open(mapping_path, "r") as f:
    class_mappings = json.load(f)

# --- SCALE ---
X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_test_scaled = X_scaler.fit_transform(X_test)
if reg_cols:
    Y_test_scaled = Y_scaler.fit_transform(Y_test[reg_cols])

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- LOAD MODEL ---
model = HybridOutputMLP(input_dim=input_dim, cls_dims=cls_dims, reg_count=reg_count, hidden_dim=64)
model.load_state_dict(torch.load(model_path))
model.eval()

# --- PREDICT ---
cls_preds = [[] for _ in cls_cols]
reg_preds = [[] for _ in reg_cols]

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Making Predictions"):
        X_batch = batch[0]
        cls_outs, reg_outs = model(X_batch)

        for i, out in enumerate(cls_outs):
            preds_idx = out.argmax(dim=1).cpu().numpy()
            mapping = class_mappings.get(cls_cols[i], {})
            # Decode class index to original label
            preds = [mapping.get(str(idx), str(idx)) for idx in preds_idx]
            cls_preds[i].extend(preds)

        for i, out in enumerate(reg_outs):
            preds = out.cpu().numpy()
            reg_preds[i].extend(preds)

# --- BUILD FINAL DATAFRAME ---
pred_dict = {}

# Classification
for i, col in enumerate(cls_cols):
    pred_dict[f"{col}_pred"] = cls_preds[i]

# Regression
if reg_cols:
    reg_preds_np = np.hstack(reg_preds).reshape(len(X_test), -1)
    reg_preds_inverse = Y_scaler.inverse_transform(reg_preds_np)
    for i, col in enumerate(reg_cols):
        pred_dict[f"{col}_pred"] = reg_preds_inverse[:, i]

# Timestamp
test_timestamps = full_df.index[split_index:]
pred_df = pd.DataFrame(pred_dict)
pred_df.insert(0, "timestamp", pd.to_datetime(test_timestamps))

# --- SAVE ---
os.makedirs(os.path.dirname(pred_path), exist_ok=True)
pred_df.to_csv(pred_path, index=False)

print(f"✅ Predictions saved to {pred_path}")
